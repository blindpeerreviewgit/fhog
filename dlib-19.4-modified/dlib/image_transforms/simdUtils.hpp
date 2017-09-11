#ifndef SIMD_UTILS_HPP
#define SIMD_UTILS_HPP 1

// 0 --> normal setting (full speed),  1 --> debug purpose only (slower)
#define MINIMISE_FLOAT_VARIATIONS 0

#include <tmmintrin.h> // SSSE3
#include <stdint.h> 

#define SIMD_VECTOR_SIZE_SHIFT 4
#define SIMD_VECTOR_SIZE       (1<<SIMD_VECTOR_SIZE_SHIFT)
#define SIMD_VECTOR_SIZE_MASK  (SIMD_VECTOR_SIZE-1)

typedef uint8_t v16qu __attribute__((__vector_size__(SIMD_VECTOR_SIZE)));
typedef int32_t v4si  __attribute__((__vector_size__(SIMD_VECTOR_SIZE)));
typedef int16_t v8hi  __attribute__((__vector_size__(SIMD_VECTOR_SIZE)));
typedef float   v4sf  __attribute__((__vector_size__(SIMD_VECTOR_SIZE)));

// https://software.intel.com/sites/landingpage/IntrinsicsGuide
// http://www.alfredklomp.com/programming/sse-intrinsics/

//----------------------------------------------------------------------------

inline
void
stream_prefetch(const void *addr,
                int byteOffset=0)
{
_mm_prefetch(((const uint8_t *)addr)+byteOffset,_MM_HINT_NTA);
}

//----------------------------------------------------------------------------

template<typename V>
inline
V
load_a(const V *aligned_addr)
{
return V(_mm_load_si128((const __m128i*)aligned_addr));
}

template<typename V>
inline
V
load_u(const V *unaligned_addr)
{
return V(_mm_loadu_si128((const __m128i*)unaligned_addr));
}

//----------------------------------------------------------------------------

template<typename V>
inline
void
store_a(V *aligned_addr,
        V v)
{
_mm_store_si128((__m128i*)aligned_addr,__m128i(v));
}

template<typename V>
inline
void
store_u(V *unaligned_addr,
        V v)
{
_mm_storeu_si128((__m128i*)unaligned_addr,__m128i(v));
}

//----------------------------------------------------------------------------

template<typename V>
inline
void
accum_a(V *aligned_addr,
        V v)
{
store_a(aligned_addr,load_a(aligned_addr)+v);
}

template<typename V>
inline
void
accum_u(V *unaligned_addr,
        V v)
{
store_u(unaligned_addr,load_u(unaligned_addr)+v);
}

//----------------------------------------------------------------------------

inline
v16qu
make_v16qu(uint8_t v)
{
return v16qu{ v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v };
}

inline
v8hi
make_v8hi(int16_t v)
{
return v8hi{ v, v, v, v, v, v, v, v };
}

inline
v4si
make_v4si(int32_t v)
{
return v4si{ v, v, v, v };
}

inline
v4sf
make_v4sf(float v)
{
return v4sf{ v, v, v, v };
}

//----------------------------------------------------------------------------

inline
v16qu
make_v16qu_0(v8hi v)
{
return v16qu(_mm_packus_epi16(__m128i(v),_mm_setzero_si128()));
}

inline
v16qu
make_v16qu_1(v8hi v)
{
return v16qu(_mm_packus_epi16(_mm_setzero_si128(),__m128i(v)));
}

inline
v16qu
make_v16qu(v8hi v0,
           v8hi v1)
{
return v16qu(_mm_packus_epi16(__m128i(v0),__m128i(v1)));
}

inline
v16qu
make_v16qu_0(v4si v)
{
const __m128i zz=_mm_setzero_si128();
return v16qu(_mm_packus_epi16(_mm_packs_epi32(__m128i(v),zz),zz));
}

inline
v16qu
make_v16qu_1(v4si v)
{
const __m128i zz=_mm_setzero_si128();
return v16qu(_mm_packus_epi16(_mm_packs_epi32(zz,__m128i(v)),zz));
}

inline
v16qu
make_v16qu_2(v4si v)
{
const __m128i zz=_mm_setzero_si128();
return v16qu(_mm_packus_epi16(zz,_mm_packs_epi32(__m128i(v),zz)));
}

inline
v16qu
make_v16qu_3(v4si v)
{
const __m128i zz=_mm_setzero_si128();
return v16qu(_mm_packus_epi16(zz,_mm_packs_epi32(zz,__m128i(v))));
}

inline
v16qu
make_v16qu(v4si v0,
           v4si v1,
           v4si v2,
           v4si v3)
{
return v16qu(_mm_packus_epi16(_mm_packs_epi32(__m128i(v0),__m128i(v1)),
                              _mm_packs_epi32(__m128i(v2),__m128i(v3))));
}

//----------------------------------------------------------------------------

inline
v8hi
make_v8hi_0u(v16qu v)
{
return v8hi(_mm_unpacklo_epi8(__m128i(v),_mm_setzero_si128()));
}

inline
v8hi
make_v8hi_0s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
return v8hi(_mm_unpacklo_epi8(__m128i(v),neg));
}

inline
v8hi
make_v8hi_1u(v16qu v)
{
return v8hi(_mm_unpackhi_epi8(__m128i(v),_mm_setzero_si128()));
}

inline
v8hi
make_v8hi_1s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
return v8hi(_mm_unpackhi_epi8(__m128i(v),neg));
}

inline
v8hi
make_v8hi_0(v4si v)
{
return v8hi(_mm_packs_epi32(__m128i(v),_mm_setzero_si128()));
}

inline
v8hi
make_v8hi_1(v4si v)
{
return v8hi(_mm_packs_epi32(_mm_setzero_si128(),__m128i(v)));
}

inline
v8hi
make_v8hi(v4si v0,
          v4si v1)
{
return v8hi(_mm_packs_epi32(__m128i(v0),__m128i(v1)));
}

//----------------------------------------------------------------------------

inline
v4si
make_v4si_0u(v16qu v)
{
const __m128i zz=_mm_setzero_si128();
return v4si(_mm_unpacklo_epi8(_mm_unpacklo_epi8(__m128i(v),zz),zz));
}

inline
v4si
make_v4si_0s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
const __m128i neg2=_mm_unpacklo_epi8(neg,neg);
return v4si(_mm_unpacklo_epi8(_mm_unpacklo_epi8(__m128i(v),neg),neg2));
}

inline
v4si
make_v4si_1u(v16qu v)
{
const __m128i zz=_mm_setzero_si128();
return v4si(_mm_unpackhi_epi8(_mm_unpacklo_epi8(__m128i(v),zz),zz));
}

inline
v4si
make_v4si_1s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
const __m128i neg2=_mm_unpacklo_epi8(neg,neg);
return v4si(_mm_unpackhi_epi8(_mm_unpacklo_epi8(__m128i(v),neg),neg2));
}

inline
v4si
make_v4si_2u(v16qu v)
{
const __m128i zz=_mm_setzero_si128();
return v4si(_mm_unpacklo_epi8(_mm_unpackhi_epi8(__m128i(v),zz),zz));
}

inline
v4si
make_v4si_2s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
const __m128i neg2=_mm_unpackhi_epi8(neg,neg);
return v4si(_mm_unpacklo_epi8(_mm_unpackhi_epi8(__m128i(v),neg),neg2));
}

inline
v4si
make_v4si_3u(v16qu v)
{
const __m128i zz=_mm_setzero_si128();
return v4si(_mm_unpackhi_epi8(_mm_unpackhi_epi8(__m128i(v),zz),zz));
}

inline
v4si
make_v4si_3s(v16qu v)
{
const __m128i neg=_mm_cmplt_epi8(__m128i(v),_mm_setzero_si128());
const __m128i neg2=_mm_unpackhi_epi8(neg,neg);
return v4si(_mm_unpackhi_epi8(_mm_unpackhi_epi8(__m128i(v),neg),neg2));
}

inline
v4si
make_v4si_0u(v8hi v)
{
return v4si(_mm_unpacklo_epi16(__m128i(v),_mm_setzero_si128()));
}

inline
v4si
make_v4si_0s(v8hi v)
{
const __m128i neg=_mm_cmplt_epi16(__m128i(v),_mm_setzero_si128());
return v4si(_mm_unpacklo_epi16(__m128i(v),neg));
}

inline
v4si
make_v4si_1u(v8hi v)
{
return v4si(_mm_unpackhi_epi16(__m128i(v),_mm_setzero_si128()));
}

inline
v4si
make_v4si_1s(v8hi v)
{
const __m128i neg=_mm_cmplt_epi16(__m128i(v),_mm_setzero_si128());
return v4si(_mm_unpackhi_epi16(__m128i(v),neg));
}

inline
v4si
make_v4si(v4sf v)
{
return v4si(_mm_cvtps_epi32(__m128(v)));
}

inline
v4sf
make_v4sf(v4si v)
{
return v4sf(_mm_cvtepi32_ps(__m128i(v)));
}

//----------------------------------------------------------------------------

#if !defined __clang__ && (__GNUC__*100+__GNUC_MINOR__)<407
  // only available starting from gcc 4.7?
# define DEFINE_VEC_AT(vType,eType)                                   \
  inline eType   vec_at(vType  v, int i) { return ((eType *)&v)[i]; } \
  inline eType & vec_at(vType &v, int i) { return ((eType *)&v)[i]; }
  DEFINE_VEC_AT(v16qu,uint8_t)
  DEFINE_VEC_AT(v8hi,uint16_t)
  DEFINE_VEC_AT(v4si,uint32_t)
  DEFINE_VEC_AT(v4sf,float)
#else
# define vec_at(v,i) ((v)[(i)])
#endif

//----------------------------------------------------------------------------

inline
v16qu
all_zeros()
{
return v16qu(_mm_setzero_si128());
}

inline
v16qu
all_ones()
{
#if !defined __clang__ && (__GNUC__*100+__GNUC_MINOR__)<407
  // only available starting from gcc 4.7?
  const __m128i undefined=undefined; // uninitilized warning will be issued!
#else
  const __m128i undefined=_mm_undefined_si128();
#endif
return v16qu(_mm_cmpeq_epi8(undefined,undefined));
}

inline
v16qu
complement(v16qu m)
{
return v16qu(_mm_xor_si128(__m128i(all_ones()),__m128i(m)));
}

template<typename T>
inline
T
filter(v16qu m,
       T v)
{
return T(_mm_and_si128(__m128i(m),__m128i(v)));
}

template<typename T>
inline
T
filter_out(v16qu m,
           T v)
{
return T(_mm_andnot_si128(__m128i(m),__m128i(v)));
}

template<typename T>
inline
T
select(v16qu m,
       T a,
       T b)
{
return T(_mm_or_si128(_mm_and_si128(__m128i(m),__m128i(a)),
                      _mm_andnot_si128(__m128i(m),__m128i(b))));
}

//----------------------------------------------------------------------------

inline
v16qu
cmp_eq(v16qu a,
       v16qu b)
{
return v16qu(_mm_cmpeq_epi8(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_gt(v16qu a,
       v16qu b)
{
return v16qu(_mm_cmpgt_epi8(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_lt(v16qu a,
       v16qu b)
{
return v16qu(_mm_cmplt_epi8(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_eq(v8hi a,
       v8hi b)
{
return v16qu(_mm_cmpeq_epi16(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_gt(v8hi a,
       v8hi b)
{
return v16qu(_mm_cmpgt_epi16(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_lt(v8hi a,
       v8hi b)
{
return v16qu(_mm_cmplt_epi16(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_eq(v4si a,
       v4si b)
{
return v16qu(_mm_cmpeq_epi32(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_gt(v4si a,
       v4si b)
{
return v16qu(_mm_cmpgt_epi32(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_lt(v4si a,
       v4si b)
{
return v16qu(_mm_cmplt_epi32(__m128i(a),__m128i(b)));
}

inline
v16qu
cmp_eq(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmpeq_ps(__m128(a),__m128(b)));
}

inline
v16qu
cmp_neq(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmpneq_ps(__m128(a),__m128(b)));
}

inline
v16qu
cmp_gt(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmpgt_ps(__m128(a),__m128(b)));
}

inline
v16qu
cmp_lt(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmplt_ps(__m128(a),__m128(b)));
}

inline
v16qu
cmp_ge(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmpge_ps(__m128(a),__m128(b)));
}

inline
v16qu
cmp_le(v4sf a,
       v4sf b)
{
return v16qu(_mm_cmple_ps(__m128(a),__m128(b)));
}

//----------------------------------------------------------------------------

inline
v16qu
min(v16qu a,
    v16qu b)
{
return v16qu(_mm_min_epu8(__m128i(a),__m128i(b)));
}

inline
v16qu
max(v16qu a,
    v16qu b)
{
return v16qu(_mm_max_epu8(__m128i(a),__m128i(b)));
}

inline
v8hi
min(v8hi a,
    v8hi b)
{
return v8hi(_mm_min_epi16(__m128i(a),__m128i(b)));
}

inline
v8hi
max(v8hi a,
    v8hi b)
{
return v8hi(_mm_max_epi16(__m128i(a),__m128i(b)));
}

inline
v4si
min(v4si a,
    v4si b)
{
return select(cmp_lt(a,b),a,b);
}

inline
v4si
max(v4si a,
    v4si b)
{
return select(cmp_gt(a,b),a,b);
}

inline
v4sf
min(v4sf a,
    v4sf b)
{
return v4sf(_mm_min_ps(__m128(a),__m128(b)));
}

inline
v4sf
max(v4sf a,
    v4sf b)
{
return v4sf(_mm_max_ps(__m128(a),__m128(b)));
}

template<typename T>
inline
T
clamp(T low,
      T high,
      T v)
{
return max(low,min(high,v));
}

//----------------------------------------------------------------------------

inline
int32_t
hsum(v4si v)
{
const __m128i a=_mm_hadd_epi32(__m128i(v),_mm_setzero_si128());
return int(_mm_cvtsi128_si32(_mm_hadd_epi32(a,_mm_setzero_si128())));
}

inline
float
hsum(v4sf v)
{
__m128 a=_mm_movehdup_ps(__m128(v)); // v3    | v3    | v1    | v1
__m128 b=_mm_add_ps(__m128(v),a);    // v3+v3 | v2+v3 | v1+v1 | v0+v1
a=_mm_movehl_ps(a,b);                // v3    | v3    | v3+v3 | v2+v3
b=_mm_add_ss(b,a);                   // v3+v3 | v2+v3 | v1+v1 | v0+v1+v2+v3
return _mm_cvtss_f32(b);             //                         v0+v1+v2+v3
}

inline
v4sf
sqrt(v4sf v)
{
return v4sf(_mm_sqrt_ps(__m128(v)));
}

inline
v16qu
abs(v16qu v)
{
return v16qu(_mm_abs_epi8(__m128i(v)));
}

inline
v8hi
abs(v8hi v)
{
return v8hi(_mm_abs_epi16(__m128i(v)));
}

inline
v4si
abs(v4si v)
{
return v4si(_mm_abs_epi32(__m128i(v)));
}

inline
v16qu
avg(v16qu a,
    v16qu b)
{
return v16qu(_mm_avg_epu8(__m128i(a),__m128i(b)));
}

inline
v8hi
avg(v8hi a,
    v8hi b)
{
return v8hi(_mm_avg_epu16(__m128i(a),__m128i(b)));
}

inline
v16qu
add_sat(v16qu a,
        v16qu b)
{
return v16qu(_mm_adds_epu8(__m128i(a),__m128i(b)));
}

inline
v8hi
mul(v8hi a,
    v8hi b)
{
return v8hi(_mm_mullo_epi16(__m128i(a),__m128i(b)));
}

inline
v4si
mul(v4si a,
    v4si b)
{
#if MINIMISE_FLOAT_VARIATIONS
  return a*b; // seems to be slower than float multiplication!
#else
  return make_v4si(make_v4sf(a)*make_v4sf(b));
#endif
}

inline
v8hi
div_u(v8hi a,
      v8hi b)
{
#if MINIMISE_FLOAT_VARIATIONS
  return a/b; // generates 8 serial scalar divisions!
#else
  const v4sf a0=make_v4sf(make_v4si_0u(a));
  const v4sf a1=make_v4sf(make_v4si_1u(a));
  const v4sf b0=make_v4sf(make_v4si_0u(b));
  const v4sf b1=make_v4sf(make_v4si_1u(b));
  return make_v8hi(make_v4si(a0/b0),make_v4si(a1/b1));
#endif
}

inline
v8hi
div_s(v8hi a,
      v8hi b)
{
#if MINIMISE_FLOAT_VARIATIONS
  return a/b; // generates 8 serial scalar divisions!
#else
  const v4sf a0=make_v4sf(make_v4si_0s(a));
  const v4sf a1=make_v4sf(make_v4si_1s(a));
  const v4sf b0=make_v4sf(make_v4si_0s(b));
  const v4sf b1=make_v4sf(make_v4si_1s(b));
  return make_v8hi(make_v4si(a0/b0),make_v4si(a1/b1));
#endif
}

inline
v4si
div(v4si a,
    v4si b)
{
#if MINIMISE_FLOAT_VARIATIONS
  return a/b; // generates 4 serial scalar divisions!
#else
  return make_v4si(make_v4sf(a)/make_v4sf(b));
#endif
}

template<int Bytes>
inline
v16qu
lmove(v16qu v)
{
return v16qu(_mm_slli_si128(__m128i(v),Bytes));
}

template<int Bytes>
inline
v16qu
rmove(v16qu v)
{
return v16qu(_mm_srli_si128(__m128i(v),Bytes));
}

template<int Bits>
inline
v4si
lshift(v4si v)
{
return v4si(_mm_slli_epi32(__m128i(v),Bits));
}

template<int Bits>
inline
v4si
rshift(v4si v)
{
return v4si(_mm_srai_epi32(__m128i(v),Bits)); // sign-extension
}

template<int Bits>
inline
v8hi
lshift(v8hi v)
{
return v8hi(_mm_slli_epi16(__m128i(v),Bits));
}

template<int Bits>
inline
v8hi
rshift(v8hi v)
{
return v8hi(_mm_srai_epi16(__m128i(v),Bits)); // sign-extension
}

template<int Bits>
inline
v16qu
lshift(v16qu v)
{
// no epi8-lshift, thus use epi16-lshift and clear low-bits
v16qu m=make_v16qu(255&~((1<<Bits)-1));
return filter(m,v16qu(lshift<Bits>(v8hi(v))));
}

template<int Bits>
inline
v16qu
rshift(v16qu v)
{
// no epi8-rshift, thus use epi16-rshift and clear high-bits
v16qu m=make_v16qu(255>>Bits);
return filter(m,v16qu(rshift<Bits>(v8hi(v))));
}

//----------------------------------------------------------------------------

inline
v16qu
interleave_0(v16qu a,
             v16qu b)
{
return v16qu(_mm_unpacklo_epi8(__m128i(a),__m128i(b)));
}

inline
v16qu
interleave_1(v16qu a,
             v16qu b)
{
return v16qu(_mm_unpackhi_epi8(__m128i(a),__m128i(b)));
}

inline
v16qu
get0of2_0(v16qu v)
{
const v16qu m=
  { 0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get0of2_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of2_0(v16qu v)
{
const v16qu m=
  { 0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of2_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get0of4_0(v16qu v)
{
const v16qu m=
  { 0x00, 0x04, 0x08, 0x0C,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get0of4_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x04, 0x08, 0x0C,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get0of4_2(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x04, 0x08, 0x0C,
    0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get0of4_3(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x04, 0x08, 0x0C };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of4_0(v16qu v)
{
const v16qu m=
  { 0x01, 0x05, 0x09, 0x0D,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of4_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0x01, 0x05, 0x09, 0x0D,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of4_2(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x01, 0x05, 0x09, 0x0D,
    0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get1of4_3(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x01, 0x05, 0x09, 0x0D };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get2of4_0(v16qu v)
{
const v16qu m=
  { 0x02, 0x06, 0x0A, 0x0E,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get2of4_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0x02, 0x06, 0x0A, 0x0E,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get2of4_2(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x02, 0x06, 0x0A, 0x0E,
    0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get2of4_3(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x02, 0x06, 0x0A, 0x0E };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get3of4_0(v16qu v)
{
const v16qu m=
  { 0x03, 0x07, 0x0B, 0x0F,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get3of4_1(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0x03, 0x07, 0x0B, 0x0F,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get3of4_2(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x03, 0x07, 0x0B, 0x0F,
    0xFF, 0xFF, 0xFF, 0xFF };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

inline
v16qu
get3of4_3(v16qu v)
{
const v16qu m=
  { 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x03, 0x07, 0x0B, 0x0F };
return v16qu(_mm_shuffle_epi8(__m128i(v),__m128i(m)));
}

#endif // SIMD_UTILS_HPP

//----------------------------------------------------------------------------
