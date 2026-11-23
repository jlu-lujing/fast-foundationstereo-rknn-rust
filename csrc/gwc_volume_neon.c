/* Group-wise correlation volume, vectorised.
 *
 * Copyright (c) 2026 Lu Jing and contributors. MIT licensed.
 *
 *   out[g,d,h,w] = sum_k L[g*K+k,h,w] * R[g*K+k,h,w-d]
 *   normalised by (||L|| * ||R|| + 1e-5) over the K = C/G channels of group g,
 *   and zero where w-d < 0.     in: [C][H][W],  out: [G][D][H][W], fp32.
 *
 * Vectorisation notes
 * -------------------
 * Disparity is the axis that is contiguous in the output, but the inputs are
 * contiguous along w and both input reads are contiguous along w.  So w is the
 * vector axis (4 fp32 lanes) and disparity is unrolled by 4 instead, letting one
 * L load feed four FMAs and taking the load/FMA ratio from 2.0 down to 1.25.
 * With K=28 this loop is load-bound long before it is FMA-bound, so that ratio
 * is the whole game.
 *
 * The w-d < 0 region would be a branch inside that loop, so it is removed
 * structurally: each R row is copied into a zero-padded window of width W+D-1
 * (as the NumPy reference does) and both norms are taken over that same window.
 * Rows keep SLACK lanes either side, because the unrolled disparity reads can
 * reach three elements before the window; they then read zeros instead of memory
 * we do not own.  Norms are square-rooted once per (g,h), so the inner loop has
 * no sqrt at all.
 *
 * Build on aarch64:
 *   gcc -O3 -march=armv8-a -fPIC -shared -o libgwcvol.so csrc/gwc_volume_neon.c
 * A scalar path is kept so the maths stays testable on any host.
 */
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define GWC_ARM 1
#else
#define GWC_ARM 0
#endif

#define SLACK 4                              /* >= 3: unrolled reads reach -3 */
#define SQRT  __builtin_sqrtf

/* One (g,h) plane, plain C: used for every column on non-ARM hosts and for the
 * few columns and disparity values the vector path deliberately leaves over. */
static void gwc_scalar(const float *Lh, const float *Rh, float *og,
                       size_t HW, int W, int K, int d, int w0, int w1)
{
    for (int w = w0; w < w1; ++w) {
        const int x = w - d;
        float acc = 0.f, ls = 0.f, rs = 0.f;
        for (int k = 0; k < K; ++k) {
            const float lv = Lh[(size_t)k * HW + w];
            const float rv = (x >= 0) ? Rh[(size_t)k * HW + x] : 0.f;
            acc += lv * rv;
            ls  += lv * lv;
            rs  += rv * rv;
        }
        (void)W;
        og[(size_t)d * HW + w] = acc / (SQRT(ls) * SQRT(rs) + 1e-5f);
    }
}

int gwc_volume_neon(const float *left, const float *right, float *out,
                    int C, int H, int W, int G, int D)
{
    if (!left || !right || !out) return -1;
    if (C <= 0 || H <= 0 || W <= 0 || G <= 0 || D <= 0 || C % G) return -1;

    const int  K  = C / G;
    const int  PW = W + D - 1;              /* padded window width   */
    const int  RS = PW + 2 * SLACK;         /* padded row stride     */
    const int  SS = PW + 2 * SLACK;         /* padded norm stride    */
    const size_t HW = (size_t)H * W;

    float *rp  = (float *)malloc((size_t)K * RS * sizeof(float));
    float *sps = (float *)malloc((size_t)SS * sizeof(float));
    float *sqs = (float *)malloc((size_t)W * sizeof(float));
    if (!rp || !sps || !sqs) { free(rp); free(sps); free(sqs); return -2; }

    for (int g = 0; g < G; ++g) {
        const float *Lg = left  + (size_t)g * K * HW;
        const float *Rg = right + (size_t)g * K * HW;
        float *og = out + (size_t)g * D * HW;

        for (int h = 0; h < H; ++h) {
            const float *Lh = Lg + (size_t)h * W;
            const float *Rh = Rg + (size_t)h * W;

            for (int k = 0; k < K; ++k) {                     /* pad this group */
                float *dst = rp + (size_t)k * RS;
                memset(dst, 0, (size_t)RS * sizeof(float));
                memcpy(dst + SLACK + (D - 1), Rh + (size_t)k * HW, (size_t)W * sizeof(float));
            }

            for (int i = 0; i < SS; ++i) sps[i] = 0.f;        /* right norm      */
            for (int k = 0; k < K; ++k) {
                const float *r = rp + (size_t)k * RS;
                int i = 0;
#if GWC_ARM
                for (; i + 4 <= SS; i += 4) {
                    float32x4_t v = vld1q_f32(r + i);
                    vst1q_f32(sps + i, vaddq_f32(vld1q_f32(sps + i), vmulq_f32(v, v)));
                }
#endif
                for (; i < SS; ++i) sps[i] += r[i] * r[i];
            }
            for (int i = 0; i < W; ++i) sqs[i] = 0.f;         /* left norm       */
            for (int k = 0; k < K; ++k) {
                const float *l = Lh + (size_t)k * HW;
                int i = 0;
#if GWC_ARM
                for (; i + 4 <= W; i += 4) {
                    float32x4_t v = vld1q_f32(l + i);
                    vst1q_f32(sqs + i, vaddq_f32(vld1q_f32(sqs + i), vmulq_f32(v, v)));
                }
#endif
                for (; i < W; ++i) sqs[i] += l[i] * l[i];
            }
            for (int i = 0; i < SS; ++i) sps[i] = SQRT(sps[i]);
            for (int i = 0; i < W;  ++i) sqs[i] = SQRT(sqs[i]);
            const float *sp = sps + SLACK;                    /* window coords   */

            /* wv = columns the vector path handles, dcov = disparity values it
             * handles; on a non-ARM host both are zero and everything falls to
             * the plain path below. */
#if GWC_ARM
            const int wv = W & ~3;
            const float32x4_t eps = vdupq_n_f32(1e-5f);
            int d0 = 0;
            for (; d0 + 4 <= D; d0 += 4) {
                for (int w = 0; w < wv; w += 4) {
                    const float *rk = rp + SLACK;             /* row 0, window   */
                    const float32x4_t nl = vld1q_f32(sqs + w);
                    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
                    for (int k = 0; k < K; ++k, rk += RS) {
                        const float32x4_t lv = vld1q_f32(Lh + (size_t)k * HW + w);
                        a0 = vfmaq_f32(a0, lv, vld1q_f32(rk + (D - 1 - d0 - 0) + w));
                        a1 = vfmaq_f32(a1, lv, vld1q_f32(rk + (D - 1 - d0 - 1) + w));
                        a2 = vfmaq_f32(a2, lv, vld1q_f32(rk + (D - 1 - d0 - 2) + w));
                        a3 = vfmaq_f32(a3, lv, vld1q_f32(rk + (D - 1 - d0 - 3) + w));
                    }
                    float *base = og + (size_t)h * W + w;
                    vst1q_f32(base + (size_t)(d0 + 0) * HW,
                              vdivq_f32(a0, vaddq_f32(vmulq_f32(nl, vld1q_f32(sp + (D - 1 - d0 - 0) + w)), eps)));
                    vst1q_f32(base + (size_t)(d0 + 1) * HW,
                              vdivq_f32(a1, vaddq_f32(vmulq_f32(nl, vld1q_f32(sp + (D - 1 - d0 - 1) + w)), eps)));
                    vst1q_f32(base + (size_t)(d0 + 2) * HW,
                              vdivq_f32(a2, vaddq_f32(vmulq_f32(nl, vld1q_f32(sp + (D - 1 - d0 - 2) + w)), eps)));
                    vst1q_f32(base + (size_t)(d0 + 3) * HW,
                              vdivq_f32(a3, vaddq_f32(vmulq_f32(nl, vld1q_f32(sp + (D - 1 - d0 - 3) + w)), eps)));
                }
            }
            const int dcov = d0;
#else
            const int wv = 0;
            const int dcov = 0;
            (void)sp;
#endif
            /* leftovers: the unaligned disparity values over every column, then
             * the unaligned columns under the disparity values already covered */
            for (int d = dcov; d < D; ++d)
                gwc_scalar(Lh, Rh, og + (size_t)h * W, HW, W, K, d, 0, W);
            for (int d = 0; d < dcov; ++d)
                gwc_scalar(Lh, Rh, og + (size_t)h * W, HW, W, K, d, wv, W);
        }
    }
    free(rp); free(sps); free(sqs);
    return 0;
}
