#ifndef FRAME_UTILS_HPP
#define FRAME_UTILS_HPP 1

#include "simdUtils.hpp"
#include <memory>
#include <cstring>
#include <vector>
#include <string>

// may be changed: 0 --> 320x240 average frame,  1 --> 640x480 full frame
#define FULL_FRAME 1

enum {
#if FULL_FRAME
       FRAME_WIDTH           =1280, //640,
       FRAME_HEIGHT          =960, //480,
#else
       FRAME_WIDTH           =320,
       FRAME_HEIGHT          =240,
#endif
       FRAME_ALIGNMENT       =16 };

static_assert(FRAME_ALIGNMENT%SIMD_VECTOR_SIZE==0,
              "FRAME_ALIGNMENT must be a multiple of SIMD_VECTOR_SIZE");

static_assert(FRAME_WIDTH%FRAME_ALIGNMENT==0,
              "FRAME_WIDTH must be a multiple of FRAME_ALIGNMENT");

enum { HISTO_CELL_SIZE       =8,
       HISTO_BIN_WIDTH_SHIFT =5,
       HISTO_BIN_WIDTH       =1<<HISTO_BIN_WIDTH_SHIFT,
       HISTO_BIN_WIDTH_MASK  =HISTO_BIN_WIDTH-1,
       HISTO_BIN_COUNT_SHIFT =8-HISTO_BIN_WIDTH_SHIFT,
       HISTO_BIN_COUNT       = 18, // 1<<HISTO_BIN_COUNT_SHIFT, // 18
       HISTO_BIN_COUNT_MASK  =HISTO_BIN_COUNT-1 };

static_assert(SIMD_VECTOR_SIZE%HISTO_CELL_SIZE==0,
              "SIMD_VECTOR_SIZE must be a multiple of HISTO_CELL_SIZE");

static_assert(256%HISTO_BIN_WIDTH==0,
              "256 must be a multiple of HISTO_BIN_WIDTH");

// static_assert(SIMD_VECTOR_SIZE%HISTO_BIN_COUNT==0,
//               "SIMD_VECTOR_SIZE must be a multiple of HISTO_BIN_COUNT");

#if 1
  // try to keep related functions close to each other
  // in order to minimise instruction cache trashing
  #define SCALAR_SECTION __attribute__ ((__section__("scalar_frame_utils")))
  #define SIMD_SECTION   __attribute__ ((__section__("simd_frame_utils")))
#else
  #define SCALAR_SECTION
  #define SIMD_SECTION
#endif

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_V4L_to_YUV(uint8_t *y_out_aligned,
                  uint8_t *u_out_aligned,
                  uint8_t *v_out_aligned,
                  const uint8_t *v4l_in_aligned,
                  int width,
                  int height);

SIMD_SECTION
void
simd_V4L_to_YUV(uint8_t *y_out_aligned,
                uint8_t *u_out_aligned,
                uint8_t *v_out_aligned,
                const uint8_t *v4l_in_aligned,
                int width,
                int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_Y_to_DX(uint8_t *dx_out_aligned,
               const uint8_t *y_in_aligned,
               int width,
               int height);

SCALAR_SECTION
void
scalar_Y_to_DY(uint8_t *dy_out_aligned,
               const uint8_t *y_in_aligned,
               int width,
               int height);

SCALAR_SECTION
void
scalar_Y_to_DXY(uint8_t *dx_out_aligned,
                uint8_t *dy_out_aligned,
                const uint8_t *y_in_aligned,
                int width,
                int height);

SIMD_SECTION
void
simd_Y_to_DX(uint8_t *dx_out_aligned,
             const uint8_t *y_in_aligned,
             int width,
             int height);

SIMD_SECTION
void
simd_Y_to_DY(uint8_t *dy_out_aligned,
             const uint8_t *y_in_aligned,
             int width,
             int height);

SIMD_SECTION
void
simd_Y_to_DXY(uint8_t *dx_out_aligned,
              uint8_t *dy_out_aligned,
              const uint8_t *y_in_aligned,
              int width,
              int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_DXY_to_MA(uint8_t *m_out_aligned,
                 uint8_t *a_out_aligned,
                 const uint8_t *dx_in_aligned,
                 const uint8_t *dy_in_aligned,
                 int width,
                 int height);

SIMD_SECTION
void
simd_DXY_to_MA(uint8_t *m_out_aligned,
               uint8_t *a_out_aligned,
               const uint8_t *dx_in_aligned,
               const uint8_t *dy_in_aligned,
               int width,
               int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_MA_to_histo(float *histo_out_aligned,
                   const uint8_t *m_in_aligned,
                   const uint8_t *a_in_aligned,
                   int width,
                   int height);

SIMD_SECTION
void
simd_MA_to_histo(float *histo_out_aligned,
                 const uint8_t *m_in_aligned,
                 const uint8_t *a_in_aligned,
                 int width,
                 int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_extract_histo(float *histo_out_aligned,
                     const float *histo_in_aligned,
                     int target_width,
                     int target_height,
                     int extract_x,
                     int extract_y,
                     int extract_width,
                     int extract_height,
                     int full_width,
                     int full_height);

SIMD_SECTION
void
simd_extract_histo(float *histo_out_aligned,
                   const float *histo_in_aligned,
                   int target_width,
                   int target_height,
                   int extract_x,
                   int extract_y,
                   int extract_width,
                   int extract_height,
                   int full_width,
                   int full_height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_equalise_histo(float *histo_out_aligned,
                      const float *histo_in_aligned,
                      int width,
                      int height,
                      int block);

SIMD_SECTION
void
simd_equalise_histo(float *histo_out_aligned,
                    const float *histo_in_aligned,
                    int width,
                    int height,
                    int block);

//----------------------------------------------------------------------------

SCALAR_SECTION
int // number of pixels with HSV-distance under the threshold
scalar_YUV_to_HSV_match(const uint8_t *y_in_aligned,
                        const uint8_t *u_in_aligned,
                        const uint8_t *v_in_aligned,
                        int x,
                        int y,
                        int width,
                        int height,
                        int full_width,
                        int full_height,
                        int hRef,
                        int sRef,
                        int vRef,
                        float hWeight,
                        float sWeight,
                        float vWeight,
                        float threshold);

SIMD_SECTION
int // number of pixels with HSV-distance under the threshold
simd_YUV_to_HSV_match(const uint8_t *y_in_aligned,
                      const uint8_t *u_in_aligned,
                      const uint8_t *v_in_aligned,
                      int x,
                      int y,
                      int width,
                      int height,
                      int full_width,
                      int full_height,
                      int hRef,
                      int sRef,
                      int vRef,
                      float hWeight,
                      float sWeight,
                      float vWeight,
                      float threshold);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_YUV_to_HSV(uint8_t *h_out_aligned,
                  uint8_t *s_out_aligned,
                  uint8_t *v_out_aligned,
                  const uint8_t *y_in_aligned,
                  const uint8_t *u_in_aligned,
                  const uint8_t *v_in_aligned,
                  int x,
                  int y,
                  int width,
                  int height,
                  int full_width,
                  int full_height);

SIMD_SECTION
void
simd_YUV_to_HSV(uint8_t *h_out_aligned,
                uint8_t *s_out_aligned,
                uint8_t *v_out_aligned,
                const uint8_t *y_in_aligned,
                const uint8_t *u_in_aligned,
                const uint8_t *v_in_aligned,
                int x,
                int y,
                int width,
                int height,
                int full_width,
                int full_height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_V4L_to_YHSV(uint8_t *y_out_aligned,
                   uint8_t *h_out_aligned,
                   uint8_t *s_out_aligned,
                   uint8_t *v_out_aligned,
                   const uint8_t *v4l_in_aligned,
                   int width,
                   int height);

SIMD_SECTION
void
simd_V4L_to_YHSV(uint8_t *y_out_aligned,
                 uint8_t *h_out_aligned,
                 uint8_t *s_out_aligned,
                 uint8_t *v_out_aligned,
                 const uint8_t *v4l_in_aligned,
                 int width,
                 int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_V4L_to_YRGB(uint8_t *y_out_aligned,
                   uint8_t *r_out_aligned,
                   uint8_t *g_out_aligned,
                   uint8_t *b_out_aligned,
                   const uint8_t *v4l_in_aligned,
                   int width,
                   int height);

SIMD_SECTION
void
simd_V4L_to_YRGB(uint8_t *y_out_aligned,
                 uint8_t *r_out_aligned,
                 uint8_t *g_out_aligned,
                 uint8_t *b_out_aligned,
                 const uint8_t *v4l_in_aligned,
                 int width,
                 int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_YUV_to_RGB(uint8_t *r_out_aligned,
                  uint8_t *g_out_aligned,
                  uint8_t *b_out_aligned,
                  const uint8_t *y_in_aligned,
                  const uint8_t *u_in_aligned,
                  const uint8_t *v_in_aligned,
                  int width,
                  int height);

SIMD_SECTION
void
simd_YUV_to_RGB(uint8_t *r_out_aligned,
                uint8_t *g_out_aligned,
                uint8_t *b_out_aligned,
                const uint8_t *y_in_aligned,
                const uint8_t *u_in_aligned,
                const uint8_t *v_in_aligned,
                int width,
                int height);

//----------------------------------------------------------------------------

SCALAR_SECTION
void
scalar_RGB_to_HSV(uint8_t *h_out_aligned,
                  uint8_t *s_out_aligned,
                  uint8_t *v_out_aligned,
                  const uint8_t *r_in_aligned,
                  const uint8_t *g_in_aligned,
                  const uint8_t *b_in_aligned,
                  int width,
                  int height);

SIMD_SECTION
void
simd_RGB_to_HSV(uint8_t *h_out_aligned,
                uint8_t *s_out_aligned,
                uint8_t *v_out_aligned,
                const uint8_t *r_in_aligned,
                const uint8_t *g_in_aligned,
                const uint8_t *b_in_aligned,
                int width,
                int height);

//----------------------------------------------------------------------------

template<typename T=uint8_t>
inline
std::unique_ptr<T[]>
storage(int count)
{
const int byteCount=int(sizeof(T))*count+FRAME_ALIGNMENT;
uint8_t *data=new uint8_t[byteCount];
std::memset(data,0,byteCount);
return std::unique_ptr<T[]>((T*)data);
}

template<typename T>
inline
T *
aligned(T *storage)
{
return (T *)((intptr_t(storage)+(FRAME_ALIGNMENT-1))&
             ~intptr_t(FRAME_ALIGNMENT-1));
}

//----------------------------------------------------------------------------

class FramePyramid
{
public:

  FramePyramid();

  FramePyramid(int minSizeX,
               int fullSizeX,
               float scaleFactorX,
               int slideX,
               int minSizeY,
               int fullSizeY,
               float scaleFactorY,
               int slideY);

  int sizeCount() const
  { return int(m_xPyramid.size()); }

  int xSize(int sizeId) const
  { return m_xPyramid[sizeId].size; }

  int ySize(int sizeId) const
  { return m_yPyramid[sizeId].size; }

  int xPosCount(int sizeId) const
  { return int(m_xPyramid[sizeId].pos.size()); }

  int xPos(int sizeId, int posId) const
  { return m_xPyramid[sizeId].pos[posId]; }

  int yPosCount(int sizeId) const
  { return int(m_yPyramid[sizeId].pos.size()); }

  int yPos(int sizeId, int posId) const
  { return m_yPyramid[sizeId].pos[posId]; }

private:

  struct SizePos { int size; std::vector<int> pos; };

  void
  m_init(std::vector<SizePos> &pyramid,
         int minSize,
         int fullSize,
         float scaleFactor,
         int slide);

  std::vector<SizePos> m_xPyramid, m_yPyramid;
};

//----------------------------------------------------------------------------

class FrameTimer
{
public:

  FrameTimer(std::string title);

  void
  tic(double startTime);

  static
  double // seconds since 1970/01/01 00:00:00 UTC
  now();

private:
  std::string m_title;
  double m_accum;
  double m_last;
  int m_count;
};

#endif // FRAME_UTILS_HPP

//----------------------------------------------------------------------------
