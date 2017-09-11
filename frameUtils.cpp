#include <dlib/image_transforms/frameUtils.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sys/time.h>

#if !defined M_PI
# define M_PI 3.14159265358979323846
#endif
#define M_PIf float(M_PI)

#if (__GNUC__*100+__GNUC_MINOR__)<407
  // only available starting from gcc 4.7
# define __builtin_assume_aligned(a,b) (a)
#endif

#define ALIGNED_PTR(type,var,addr) \
  type * __restrict__ var=         \
  (type *)__builtin_assume_aligned(addr,FRAME_ALIGNMENT)

inline
void
enterSimd()
{
// adjust rounding mode in order to minimise differences
// with scalar code while converting from v4sf to v4si
_MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
}

//----------------------------------------------------------------------------

void
scalar_V4L_to_YUV(uint8_t *y_out_aligned,
                  uint8_t *u_out_aligned,
                  uint8_t *v_out_aligned,
                  const uint8_t *v4l_in_aligned,
                  int width,
                  int height)
{
ALIGNED_PTR( uint8_t, dstY, y_out_aligned );
ALIGNED_PTR( uint8_t, dstU, u_out_aligned );
ALIGNED_PTR( uint8_t, dstV, v_out_aligned );
#if FULL_FRAME
  ALIGNED_PTR( const uint8_t, src, v4l_in_aligned );
  const int pixelCount=width*height;
  for(int id=0;id<pixelCount;++id)
    {
    dstY[id]=uint8_t(src[2*id]);
    dstU[id]=uint8_t(src[((2*id)&~3)|1]);
    dstV[id]=uint8_t(src[((2*id)&~3)|3]);
    }
#else
  ALIGNED_PTR( const uint8_t, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const uint8_t, src1, src0+width*4 );
    for(int xId=0;xId<width;++xId)
      {
      // emulate _mm_avg_epu8(a,b) as (a+b+1)>>1
      const int y0=(src0[4*xId+0]+src1[4*xId+0]+1)>>1;
      const int u1=(src0[4*xId+1]+src1[4*xId+1]+1)>>1;
      const int y2=(src0[4*xId+2]+src1[4*xId+2]+1)>>1;
      const int v3=(src0[4*xId+3]+src1[4*xId+3]+1)>>1;
      const int y02=(y0+y2+1)>>1;
      dstY[xId]=uint8_t(y02);
      dstU[xId]=uint8_t(u1);
      dstV[xId]=uint8_t(v3);
      }
    src0+=width*8;
    dstY+=width;
    dstU+=width;
    dstV+=width;
    }
#endif
}

void
simd_V4L_to_YUV(uint8_t *y_out_aligned,
                uint8_t *u_out_aligned,
                uint8_t *v_out_aligned,
                const uint8_t *v4l_in_aligned,
                int width,
                int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstY, y_out_aligned );
ALIGNED_PTR( v16qu, dstU, u_out_aligned );
ALIGNED_PTR( v16qu, dstV, v_out_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
#if FULL_FRAME
  ALIGNED_PTR( const v16qu, src, v4l_in_aligned );
  const int vPixelCount=vWidth*height;
  for(int id=0;id<vPixelCount;++id)
    {
    v16qu yuyv,yy,uu,vv;
    yuyv=load_a(src+2*id+0);
    yy=get0of2_0(yuyv);
    uu=get1of4_0(yuyv);
    vv=get3of4_0(yuyv);
    yuyv=load_a(src+2*id+1);
    yy|=get0of2_1(yuyv); store_a(dstY+id,yy);
    uu|=get1of4_1(yuyv); store_a(dstU+id,interleave_0(uu,uu));
    vv|=get3of4_1(yuyv); store_a(dstV+id,interleave_0(vv,vv));
    }
#else
  ALIGNED_PTR( const v16qu, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const v16qu, src1, src0+vWidth*4 );
    for(int xId=0;xId<vWidth;++xId)
      {
      v16qu yuyv,yy,uu,vv;
      yuyv=avg(load_a(src0+4*xId+0),load_a(src1+4*xId+0));
      yy=avg(get0of4_0(yuyv),get2of4_0(yuyv));
      uu=get1of4_0(yuyv);
      vv=get3of4_0(yuyv);
      yuyv=avg(load_a(src0+4*xId+1),load_a(src1+4*xId+1));
      yy|=avg(get0of4_1(yuyv),get2of4_1(yuyv));
      uu|=get1of4_1(yuyv);
      vv|=get3of4_1(yuyv);
      yuyv=avg(load_a(src0+4*xId+2),load_a(src1+4*xId+2));
      yy|=avg(get0of4_2(yuyv),get2of4_2(yuyv));
      uu|=get1of4_2(yuyv);
      vv|=get3of4_2(yuyv);
      yuyv=avg(load_a(src0+4*xId+3),load_a(src1+4*xId+3));
      yy|=avg(get0of4_3(yuyv),get2of4_3(yuyv)); store_a(dstY+xId,yy);
      uu|=get1of4_3(yuyv);                      store_a(dstU+xId,uu);
      vv|=get3of4_3(yuyv);                      store_a(dstV+xId,vv);
      }
    src0+=vWidth*8;
    dstY+=vWidth;
    dstU+=vWidth;
    dstV+=vWidth;
    }
#endif
}

//----------------------------------------------------------------------------

void
scalar_Y_to_DX(uint8_t *dx_out_aligned,
               const uint8_t *y_in_aligned,
               int width,
               int height)
{
ALIGNED_PTR( uint8_t, dst, dx_out_aligned );
ALIGNED_PTR( const uint8_t, src, y_in_aligned );
for(int yId=0;yId<height;++yId)
  {
  int x=src[0]>>1, x0=x;
  for(int xId=0;xId<width;++xId)
    {
    const int x1=(xId==width-1) ? x : (src[xId+1]>>1);
    dst[xId]=uint8_t(128+x1-x0);
    x0=x; x=x1;
    }
  src+=width;
  dst+=width;
  }
}

void
scalar_Y_to_DY(uint8_t *dy_out_aligned,
               const uint8_t *y_in_aligned,
               int width,
               int height)
{
ALIGNED_PTR( uint8_t, dst, dy_out_aligned );
ALIGNED_PTR( const uint8_t, src0, y_in_aligned );
for(int yId=0;yId<height;++yId)
  {
  const int yStep=(yId&&(yId<height-1)) ? 2 : 1;
  ALIGNED_PTR( const uint8_t, src1, src0+width*yStep );
  for(int xId=0;xId<width;++xId)
    {
    const int y0=src0[xId]>>1, y1=src1[xId]>>1;
    dst[xId]=uint8_t(128+y1-y0);
    }
  if(yId) { src0+=width; }
  dst+=width;
  }
}

void
scalar_Y_to_DXY(uint8_t *dx_out_aligned,
                uint8_t *dy_out_aligned,
                const uint8_t *y_in_aligned,
                int width,
                int height)
{
scalar_Y_to_DX(dx_out_aligned,y_in_aligned,width,height);
scalar_Y_to_DY(dy_out_aligned,y_in_aligned,width,height);
}

void
simd_Y_to_DX(uint8_t *dx_out_aligned,
             const uint8_t *y_in_aligned,
             int width,
             int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
const v16qu offset=make_v16qu(128);
ALIGNED_PTR( v16qu, dst, dx_out_aligned );
ALIGNED_PTR( const v16qu, src, y_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
for(int yId=0;yId<height;++yId)
  {
  v16qu x=rshift<1>(load_a(src));
  v16qu x0=lmove<15>(x);
  for(int xId=0;xId<vWidth;++xId)
    {
    const v16qu x1=(xId==vWidth-1)
                   ? rmove<15>(x) : rshift<1>(load_a(src+xId+1));
    dst[xId]=offset+(lmove<15>(x1)|rmove<1>(x))-(rmove<15>(x0)|lmove<1>(x));
    x0=x; x=x1;
    }
  src+=vWidth;
  dst+=vWidth;
  }
}

void
simd_Y_to_DY(uint8_t *dy_out_aligned,
             const uint8_t *y_in_aligned,
             int width,
             int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
const v16qu offset=make_v16qu(128);
ALIGNED_PTR( v16qu, dst, dy_out_aligned );
ALIGNED_PTR( const v16qu, src0, y_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
for(int yId=0;yId<height;++yId)
  {
  const int yStep=(yId&&(yId<height-1)) ? 2 : 1;
  ALIGNED_PTR( const v16qu, src1, src0+vWidth*yStep );
  for(int xId=0;xId<vWidth;++xId)
    {
    const v16qu y0=rshift<1>(load_a(src0+xId));
    const v16qu y1=rshift<1>(load_a(src1+xId));
    store_a(dst+xId,offset+y1-y0);
    }
  if(yId) { src0+=vWidth; }
  dst+=vWidth;
  }
}

void
simd_Y_to_DXY(uint8_t *dx_out_aligned,
              uint8_t *dy_out_aligned,
              const uint8_t *y_in_aligned,
              int width,
              int height)
{
simd_Y_to_DX(dx_out_aligned,y_in_aligned,width,height);
simd_Y_to_DY(dy_out_aligned,y_in_aligned,width,height);
}

//----------------------------------------------------------------------------

void
scalar_DXY_to_MA(uint8_t *m_out_aligned,
                 uint8_t *a_out_aligned,
                 const uint8_t *dx_in_aligned,
                 const uint8_t *dy_in_aligned,
                 int width,
                 int height)
{
ALIGNED_PTR( uint8_t, dstM, m_out_aligned );
ALIGNED_PTR( uint8_t, dstA, a_out_aligned );
ALIGNED_PTR( const uint8_t, srcDx, dx_in_aligned );
ALIGNED_PTR( const uint8_t, srcDy, dy_in_aligned );
const int pixelCount=width*height;
for(int id=0;id<pixelCount;++id)
  {
  const int dx=srcDx[id]-128, dy=srcDy[id]-128;
  const int x=std::abs(dx), y=std::abs(dy);
  // approximate sqrt(x*x+y*y) with max(x,y)+min(x,y)/2
  const int twiceModule=x>y ? (x<<1)+y : (y<<1)+x;
  dstM[id]=uint8_t(std::min(255,twiceModule));
  // raw atan2() estimation: (pi/4)*y/x for x>y, then symetries
  // ( http://www.romanblack.com/integer_degree.htm )
  int cA,cB,cC;
  if(y<x) { cA=0;     cB=+(256/4)*y; cC=x; }
  else    { cA=256/2; cB=-(256/4)*x; cC=y; }
  int angle=cA+cB/std::max(1,cC);
  if((dx==x)!=(dy==y))
    { angle=-angle; }
  dstA[id]=uint8_t((angle+256)&255);
  }
}

void
simd_DXY_to_MA(uint8_t *m_out_aligned,
               uint8_t *a_out_aligned,
               const uint8_t *dx_in_aligned,
               const uint8_t *dy_in_aligned,
               int width,
               int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
const v16qu offset=make_v16qu(128);
const v8hi cOne=make_v8hi(1);
const v8hi c256=make_v8hi(256);
const v8hi c255=make_v8hi(255);
const v8hi c256_2=make_v8hi(256/2);
ALIGNED_PTR( v16qu, dstM, m_out_aligned );
ALIGNED_PTR( v16qu, dstA, a_out_aligned );
ALIGNED_PTR( const v16qu, srcDx, dx_in_aligned );
ALIGNED_PTR( const v16qu, srcDy, dy_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
const int vPixelCount=vWidth*height;
for(int id=0;id<vPixelCount;++id)
  {
  const v16qu dx=load_a(srcDx+id)-offset,
              dy=load_a(srcDy+id)-offset; // underflow is OK
  const v16qu x=abs(dx), y=abs(dy);
  // approximate sqrt(x*x+y*y) with max(x,y)+min(x,y)/2
  const v16qu twX=lshift<1>(x), twY=lshift<1>(y);
  const v16qu twiceModule=select(cmp_lt(y,x),add_sat(twX,y),add_sat(twY,x));
  store_a(dstM+id,twiceModule);
  // raw atan2() estimation: (pi/4)*y/x for x>y, then symetries
  // ( http://www.romanblack.com/integer_degree.htm )
  const v8hi dx0=make_v8hi_0s(dx), dy0=make_v8hi_0s(dy);
  const v8hi x0=make_v8hi_0u(x), y0=make_v8hi_0u(y);
  const v16qu y0_lt_x0=cmp_lt(y0,x0);
  const v8hi cA0=select(y0_lt_x0,v8hi(all_zeros()),c256_2);
  const v8hi cB0=select(y0_lt_x0,lshift<6>(y0),-lshift<6>(x0)); // 256/4
  const v8hi cC0=select(y0_lt_x0,x0,y0);
  v8hi angle0=cA0+div_s(cB0,max(cOne,cC0));
  angle0=select(cmp_eq(dx0,x0)^cmp_eq(dy0,y0),-angle0,angle0);
  angle0=(angle0+c256)&c255;
  const v8hi dx1=make_v8hi_1s(dx), dy1=make_v8hi_1s(dy);
  const v8hi x1=make_v8hi_1u(x), y1=make_v8hi_1u(y);
  const v16qu y1_lt_x1=cmp_lt(y1,x1);
  const v8hi cA1=select(y1_lt_x1,v8hi(all_zeros()),c256_2);
  const v8hi cB1=select(y1_lt_x1,lshift<6>(y1),-lshift<6>(x1)); // 256/4
  const v8hi cC1=select(y1_lt_x1,x1,y1);
  v8hi angle1=cA1+div_s(cB1,max(cOne,cC1));
  angle1=select(cmp_eq(dx1,x1)^cmp_eq(dy1,y1),-angle1,angle1);
  angle1=(angle1+c256)&c255;
  store_a(dstA+id,make_v16qu(angle0,angle1));
  }
}

//----------------------------------------------------------------------------

void
scalar_MA_to_histo(float *histo_out_aligned,
                   const uint8_t *m_in_aligned,
                   const uint8_t *a_in_aligned,
                   int width,
                   int height)
{
ALIGNED_PTR( float, dstHisto, histo_out_aligned );
ALIGNED_PTR( const uint8_t, srcM0, m_in_aligned );
ALIGNED_PTR( const uint8_t, srcA0, a_in_aligned );
for(int yId=0;yId<height;yId+=HISTO_CELL_SIZE)
  {
  for(int xId=0;xId<width;xId+=HISTO_CELL_SIZE)
    {
    ALIGNED_PTR( const uint8_t, srcM1, srcM0 );
    ALIGNED_PTR( const uint8_t, srcA1, srcA0 );
    float histo[HISTO_BIN_COUNT]={0.0f};
    for(int yi=0;yi<HISTO_CELL_SIZE;++yi)
      {
      for(int xi=0;xi<HISTO_CELL_SIZE;++xi)
        {
        const int id=xId+xi;
        const int m=srcM1[id];
        const int a=srcA1[id];
        const int binA=a>>HISTO_BIN_WIDTH_SHIFT;
        const int da=(a&HISTO_BIN_WIDTH_MASK)+(HISTO_BIN_WIDTH>>1);
        const bool low=da<HISTO_BIN_WIDTH;
        const int binB=(binA+(low ? HISTO_BIN_COUNT_MASK : 1))
                       &HISTO_BIN_COUNT_MASK;
        const float top=float(HISTO_BIN_WIDTH+HISTO_BIN_WIDTH_MASK)/
                        float(HISTO_BIN_WIDTH_MASK);
        const float x=float(da)/float(HISTO_BIN_WIDTH_MASK);
        const float hA=float(m)*(low ? x : (top-x));
        const float hB=float(m)-hA;
        histo[binA]+=hA;
        histo[binB]+=hB;
        }
      srcM1+=width;
      srcA1+=width;
      }
    for(int i=0;i<HISTO_BIN_COUNT;++i)
      { dstHisto[i]=histo[i]; }
    dstHisto+=HISTO_BIN_COUNT;
    }
  srcM0+=width*HISTO_CELL_SIZE;
  srcA0+=width*HISTO_CELL_SIZE;
  }
}

void
simd_MA_to_histo(float *histo_out_aligned,
                 const uint8_t *m_in_aligned,
                 const uint8_t *a_in_aligned,
                 int width,
                 int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
assert(SIMD_VECTOR_SIZE==2*HISTO_BIN_COUNT);
assert(SIMD_VECTOR_SIZE==2*HISTO_CELL_SIZE);
enterSimd();
ALIGNED_PTR( v4sf, dstHisto, histo_out_aligned );
ALIGNED_PTR( const v16qu, srcM0, m_in_aligned );
ALIGNED_PTR( const v16qu, srcA0, a_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
for(int yId=0;yId<height;yId+=HISTO_CELL_SIZE)
  {
  for(int xId=0;xId<vWidth;++xId)
    {
    ALIGNED_PTR( const v16qu, srcM1, srcM0 );
    ALIGNED_PTR( const v16qu, srcA1, srcA0 );
    __attribute__((__aligned__(SIMD_VECTOR_SIZE)))
      union { v4sf v[4]; float f[16]; } histo;
    histo.v[0]=histo.v[1]=histo.v[2]=histo.v[3]=v4sf(all_zeros());
    for(int yi=0;yi<HISTO_CELL_SIZE;++yi)
      {
      const v16qu m=load_a(srcM1+xId), a=load_a(srcA1+xId);
      const v16qu binA=rshift<HISTO_BIN_WIDTH_SHIFT>(a);
      const v16qu da=(a&make_v16qu(HISTO_BIN_WIDTH_MASK))+
                     make_v16qu(HISTO_BIN_WIDTH>>1);
      const v16qu low=cmp_lt(da,make_v16qu(HISTO_BIN_WIDTH));
      const v16qu binB=(binA+select(low,make_v16qu(HISTO_BIN_COUNT_MASK),
                                        make_v16qu(1)))
                       &make_v16qu(HISTO_BIN_COUNT_MASK);
      const v4sf div=make_v4sf(1.0f/float(HISTO_BIN_WIDTH_MASK));
      const v4sf top=make_v4sf(float(HISTO_BIN_WIDTH+HISTO_BIN_WIDTH_MASK)/
                               float(HISTO_BIN_WIDTH_MASK));
      const v4sf x0=make_v4sf(make_v4si_0u(da))*div,
                 m0=make_v4sf(make_v4si_0u(m)),
                 hA0=m0*select(v16qu(make_v4si_0s(low)),x0,top-x0),
                 hB0=m0-hA0;
      const v4sf x1=make_v4sf(make_v4si_1u(da))*div,
                 m1=make_v4sf(make_v4si_1u(m)),
                 hA1=m1*select(v16qu(make_v4si_1s(low)),x1,top-x1),
                 hB1=m1-hA1;
      const v4sf x2=make_v4sf(make_v4si_2u(da))*div,
                 m2=make_v4sf(make_v4si_2u(m)),
                 hA2=m2*select(v16qu(make_v4si_2s(low)),x2,top-x2),
                 hB2=m2-hA2;
      const v4sf x3=make_v4sf(make_v4si_3u(da))*div,
                 m3=make_v4sf(make_v4si_3u(m)),
                 hA3=m3*select(v16qu(make_v4si_3s(low)),x3,top-x3),
                 hB3=m3-hA3;
      for(int i=0;i<4;++i)
        {
        // this ugly serial loop is inherent to the histogram problem!
        histo.f[0+vec_at(binA, 0+i)]+=vec_at(hA0,i);
        histo.f[0+vec_at(binB, 0+i)]+=vec_at(hB0,i);
        histo.f[0+vec_at(binA, 4+i)]+=vec_at(hA1,i);
        histo.f[0+vec_at(binB, 4+i)]+=vec_at(hB1,i);
        histo.f[8+vec_at(binA, 8+i)]+=vec_at(hA2,i);
        histo.f[8+vec_at(binB, 8+i)]+=vec_at(hB2,i);
        histo.f[8+vec_at(binA,12+i)]+=vec_at(hA3,i);
        histo.f[8+vec_at(binB,12+i)]+=vec_at(hB3,i);
        }
      srcM1+=vWidth;
      srcA1+=vWidth;
      }
    store_a(dstHisto+0,load_a(histo.v+0));
    store_a(dstHisto+1,load_a(histo.v+1));
    store_a(dstHisto+2,load_a(histo.v+2));
    store_a(dstHisto+3,load_a(histo.v+3));
    dstHisto+=4;
    }
  srcM0+=vWidth*HISTO_CELL_SIZE;
  srcA0+=vWidth*HISTO_CELL_SIZE;
  }
}

//----------------------------------------------------------------------------

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
                     int full_height)
{
(void)full_height; // unused actually!
assert(extract_width>=target_width);
assert(extract_height>=target_height);
assert(extract_x+extract_width<=full_width);
assert(extract_y+extract_height<=full_height);
ALIGNED_PTR( float, dstHisto, histo_out_aligned );
ALIGNED_PTR( const float, srcHisto, histo_in_aligned );
const float scaleX=float(target_width)/float(extract_width),
            invScaleX=float(extract_width)/float(target_width);
const float scaleY=float(target_height)/float(extract_height),
            invScaleY=float(extract_height)/float(target_height);
std::memset(dstHisto,0,target_width*target_height*
                       HISTO_BIN_COUNT*sizeof(float));
for(int yId=0;yId<extract_height;++yId)
  {
  int srcId=((extract_y+yId)*full_width+extract_x)*HISTO_BIN_COUNT;
  const float yA=float(yId)*scaleY, yB=yA+scaleY;
  const int tyA=int(yA), tyB=tyA+1;
  float remainingY=yB-float(tyB);
  if((remainingY>0.0f)&&(tyB<target_height))
    {
    remainingY*=invScaleY;
    const float complementY=1.0f-remainingY;
    for(int xId=0;xId<extract_width;++xId)
      {
      const float xA=float(xId)*scaleX, xB=xA+scaleX;
      const int txA=int(xA), txB=txA+1;
      const int dstIdAA=(tyA*target_width+txA)*HISTO_BIN_COUNT;
      const int dstIdBA=(tyB*target_width+txA)*HISTO_BIN_COUNT;
      float remainingX=xB-float(txB);
      if((remainingX>0.0f)&&(txB<target_width))
        {
        remainingX*=invScaleX;
        const float complementX=1.0f-remainingX;
        const int dstIdAB=(tyA*target_width+txB)*HISTO_BIN_COUNT;
        const int dstIdBB=(tyB*target_width+txB)*HISTO_BIN_COUNT;
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float value=srcHisto[srcId+hId];
          dstHisto[dstIdAA+hId]+=value*complementY*complementX;
          dstHisto[dstIdBA+hId]+=value*remainingY*complementX;
          dstHisto[dstIdAB+hId]+=value*complementY*remainingX;
          dstHisto[dstIdBB+hId]+=value*remainingY*remainingX;
          }
        }
      else
        {
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float value=srcHisto[srcId+hId];
          dstHisto[dstIdAA+hId]+=value*complementY;
          dstHisto[dstIdBA+hId]+=value*remainingY;
          }
        }
      srcId+=HISTO_BIN_COUNT;
      }
    }
  else
    {
    for(int xId=0;xId<extract_width;++xId)
      {
      const float xA=float(xId)*scaleX, xB=xA+scaleX;
      const int txA=int(xA), txB=txA+1;
      const int dstIdAA=(tyA*target_width+txA)*HISTO_BIN_COUNT;
      float remainingX=xB-float(txB);
      if((remainingX>0.0f)&&(txB<target_width))
        {
        remainingX*=invScaleX;
        const float complementX=1.0f-remainingX;
        const int dstIdAB=(tyA*target_width+txB)*HISTO_BIN_COUNT;
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float value=srcHisto[srcId+hId];
          dstHisto[dstIdAA+hId]+=value*complementX;
          dstHisto[dstIdAB+hId]+=value*remainingX;
          }
        }
      else
        {
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float value=srcHisto[srcId+hId];
          dstHisto[dstIdAA+hId]+=value;
          }
        }
      srcId+=HISTO_BIN_COUNT;
      }
    }
  }
}

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
                   int full_height)
{
(void)full_height; // unused actually!
assert(extract_width>=target_width);
assert(extract_height>=target_height);
assert(extract_x+extract_width<=full_width);
assert(extract_y+extract_height<=full_height);
enterSimd();
ALIGNED_PTR( v4sf, dstHisto, histo_out_aligned );
ALIGNED_PTR( const v4sf, srcHisto, histo_in_aligned );
const float scaleX=float(target_width)/float(extract_width),
            invScaleX=float(extract_width)/float(target_width);
const float scaleY=float(target_height)/float(extract_height),
            invScaleY=float(extract_height)/float(target_height);
const int vBinCount=HISTO_BIN_COUNT/(sizeof(v4sf)/sizeof(float));
std::memset(dstHisto,0,target_width*target_height*
                       vBinCount*sizeof(v4sf));
for(int yId=0;yId<extract_height;++yId)
  {
  int srcId=((extract_y+yId)*full_width+extract_x)*vBinCount;
  const float yA=float(yId)*scaleY, yB=yA+scaleY;
  const int tyA=int(yA), tyB=tyA+1;
  float remainingY=yB-float(tyB);
  if((remainingY>0.0f)&&(tyB<target_height))
    {
    remainingY*=invScaleY;
    const float complementY=1.0f-remainingY;
    const v4sf vRemainingY=make_v4sf(remainingY);
    const v4sf vComplementY=make_v4sf(complementY);
    for(int xId=0;xId<extract_width;++xId)
      {
      const float xA=float(xId)*scaleX, xB=xA+scaleX;
      const int txA=int(xA), txB=txA+1;
      const int dstIdAA=(tyA*target_width+txA)*vBinCount;
      const int dstIdBA=(tyB*target_width+txA)*vBinCount;
      float remainingX=xB-float(txB);
      if((remainingX>0.0f)&&(txB<target_width))
        {
        remainingX*=invScaleX;
        const float complementX=1.0f-remainingX;
        const v4sf vRemainingX=make_v4sf(remainingX);
        const v4sf vComplementX=make_v4sf(complementX);
        const int dstIdAB=(tyA*target_width+txB)*vBinCount;
        const int dstIdBB=(tyB*target_width+txB)*vBinCount;
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf value=load_a(srcHisto+srcId+hId);
          accum_a(dstHisto+dstIdAA+hId,value*vComplementY*vComplementX);
          accum_a(dstHisto+dstIdBA+hId,value*vRemainingY*vComplementX);
          accum_a(dstHisto+dstIdAB+hId,value*vComplementY*vRemainingX);
          accum_a(dstHisto+dstIdBB+hId,value*vRemainingY*vRemainingX);
          }
        }
      else
        {
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf value=load_a(srcHisto+srcId+hId);
          accum_a(dstHisto+dstIdAA+hId,value*vComplementY);
          accum_a(dstHisto+dstIdBA+hId,value*vRemainingY);
          }
        }
      srcId+=vBinCount;
      }
    }
  else
    {
    for(int xId=0;xId<extract_width;++xId)
      {
      const float xA=float(xId)*scaleX, xB=xA+scaleX;
      const int txA=int(xA), txB=txA+1;
      const int dstIdAA=(tyA*target_width+txA)*vBinCount;
      float remainingX=xB-float(txB);
      if((remainingX>0.0f)&&(txB<target_width))
        {
        remainingX*=invScaleX;
        const float complementX=1.0f-remainingX;
        const v4sf vRemainingX=make_v4sf(remainingX);
        const v4sf vComplementX=make_v4sf(complementX);
        const int dstIdAB=(tyA*target_width+txB)*vBinCount;
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf value=load_a(srcHisto+srcId+hId);
          accum_a(dstHisto+dstIdAA+hId,value*vComplementX);
          accum_a(dstHisto+dstIdAB+hId,value*vRemainingX);
          }
        }
      else
        {
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf value=load_a(srcHisto+srcId+hId);
          accum_a(dstHisto+dstIdAA+hId,value);
          }
        }
      srcId+=vBinCount;
      }
    }
  }
}

//----------------------------------------------------------------------------

void
scalar_equalise_histo(float *histo_out_aligned,
                      const float *histo_in_aligned,
                      int width,
                      int height,
                      int block)
{
assert(block<=width);
assert(block<=height);
ALIGNED_PTR( float, dstHisto, histo_out_aligned );
ALIGNED_PTR( const float, srcHisto, histo_in_aligned );
const int target_width=width+1-block;
const int target_height=height+1-block;
int dstId=0;
for(int yId=0;yId<target_height;++yId)
  {
  for(int xId=0;xId<target_width;++xId)
    {
    float sqrSum=0.0f;
    for(int dy=0;dy<block;++dy)
      {
      for(int dx=0;dx<block;++dx)
        {
        const int srcId=((yId+dy)*width+(xId+dx))<<HISTO_BIN_COUNT_SHIFT;
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float h=srcHisto[srcId+hId];
          sqrSum+=h*h;
          }
        }
      }
    const float norm=sqrSum ? 1.0f/std::sqrt(sqrSum) : 1.0f;
    for(int dy=0;dy<block;++dy)
      {
      for(int dx=0;dx<block;++dx)
        {
        const int srcId=((yId+dy)*width+(xId+dx))<<HISTO_BIN_COUNT_SHIFT;
        for(int hId=0;hId<HISTO_BIN_COUNT;++hId)
          {
          const float h=srcHisto[srcId+hId];
          dstHisto[dstId++]=h*norm;
          }
        }
      }
    }
  }
}

void
simd_equalise_histo(float *histo_out_aligned,
                    const float *histo_in_aligned,
                    int width,
                    int height,
                    int block)
{
assert(block<=width);
assert(block<=height);
enterSimd();
ALIGNED_PTR( v4sf, dstHisto, histo_out_aligned );
ALIGNED_PTR( const v4sf, srcHisto, histo_in_aligned );
const int vBinCount=HISTO_BIN_COUNT/(sizeof(v4sf)/sizeof(float));
const int target_width=width+1-block;
const int target_height=height+1-block;
int dstId=0;
for(int yId=0;yId<target_height;++yId)
  {
  for(int xId=0;xId<target_width;++xId)
    {
    v4sf vSqrSum=v4sf(all_zeros());
    for(int dy=0;dy<block;++dy)
      {
      for(int dx=0;dx<block;++dx)
        {
        const int srcId=((yId+dy)*width+(xId+dx))*vBinCount;
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf h=load_a(srcHisto+srcId+hId);
          vSqrSum+=h*h;
          }
        }
      }
    const float sqrSum=hsum(vSqrSum);
    const v4sf norm=make_v4sf(sqrSum ? 1.0f/std::sqrt(sqrSum) : 1.0f);
    for(int dy=0;dy<block;++dy)
      {
      for(int dx=0;dx<block;++dx)
        {
        const int srcId=((yId+dy)*width+(xId+dx))*vBinCount;
        for(int hId=0;hId<vBinCount;++hId)
          {
          const v4sf h=load_a(srcHisto+srcId+hId);
          store_a(dstHisto+(dstId++),h*norm);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------

inline
void
scalar_yuv_to_rgb(int &out_r,int &out_g,int &out_b,
                  int y,int u,int v)
{
#if MINIMISE_FLOAT_VARIATIONS
  y-=16;
  u-=128;
  v-=128;
  out_r=std::max(0,std::min(255,(298*y      +409*v+128)>>8));
  out_g=std::max(0,std::min(255,(298*y-100*u-208*v+128)>>8));
  out_b=std::max(0,std::min(255,(298*y+516*u      +128)>>8));
#else
  const float yf=1.1640625f*float(y-16), uf=float(u-128), vf=float(v-128);
  const float c0=1.59765625f, c1=-0.390625f, c2=-0.8125f, c3=2.015625f;
  out_r=int(std::max(0.0f,std::min(255.0f,yf      +c0*vf)));
  out_g=int(std::max(0.0f,std::min(255.0f,yf+c1*uf+c2*vf)));
  out_b=int(std::max(0.0f,std::min(255.0f,yf+c3*uf      )));
#endif
}

inline
void
simd_yuv_to_rgb(v4si &out_r, v4si &out_g, v4si &out_b,
                v4si y, v4si u, v4si v)
{
const v4si low=v4si(all_zeros());
const v4si high=make_v4si(255);
const v4si mid=make_v4si(128);
const v4si yi=y-make_v4si(16);
const v4si ui=u-mid;
const v4si vi=v-mid;
#if MINIMISE_FLOAT_VARIATIONS
  const v4si y298p128=mul(yi,make_v4si(298))+make_v4si(128);
  const v4si u100=mul(ui,make_v4si(100));
  const v4si u516=mul(ui,make_v4si(516));
  const v4si v409=mul(vi,make_v4si(409));
  const v4si v208=mul(vi,make_v4si(208));
  out_r=clamp(low,high,rshift<8>(y298p128     +v409));
  out_g=clamp(low,high,rshift<8>(y298p128-u100-v208));
  out_b=clamp(low,high,rshift<8>(y298p128+u516     ));
#else
  const v4sf p0=make_v4sf(1.59765625f);
  const v4sf p1=make_v4sf(-0.390625f);
  const v4sf p2=make_v4sf(-0.8125f);
  const v4sf p3=make_v4sf(2.015625f);
  const v4sf yf=make_v4sf(yi)*make_v4sf(1.1640625f);
  const v4sf uf=make_v4sf(ui);
  const v4sf vf=make_v4sf(vi);
  out_r=clamp(low,high,make_v4si(yf      +p0*vf));
  out_g=clamp(low,high,make_v4si(yf+p1*uf+p2*vf));
  out_b=clamp(low,high,make_v4si(yf+p3*uf      ));
#endif
}

inline
void
scalar_rgb_to_hsv(int &out_h,int &out_s,int &out_v,
                  int r, int g, int b)
{
const int maxC=std::max(r,std::max(g,b));
const int minC=std::min(r,std::min(g,b));
const int d=std::max(1,(maxC-minC)*3);
const int h=( r==maxC ?   0+((g-b)*128)/d
            : g==maxC ?  85+((b-r)*128)/d
                      : 170+((r-g)*128)/d );
out_h=(h+256)&255;
// out_s=85*d/std::max(1,maxC);
out_s=(85/2)*d/std::max(1,maxC>>1);
out_v=maxC;
}

inline
void
simd_rgb_to_hsv(v8hi &out_h, v8hi &out_s, v8hi &out_v,
                v8hi r, v8hi g, v8hi b)
{
const v8hi maxC=max(r,max(g,b));
const v8hi minC=min(r,min(g,b));
const v16qu rMax=cmp_eq(maxC,r);
const v16qu gMax=filter_out(rMax,cmp_eq(maxC,g));
const v16qu bMax=complement(rMax|gMax);
const v8hi d=max(make_v8hi(1),mul((maxC-minC),make_v8hi(3)));
const v8hi cA=filter(gMax,make_v8hi(85))|filter(bMax,make_v8hi(170));
const v8hi cB=filter(rMax,g)|filter(gMax,b)|filter(bMax,r);
const v8hi cC=filter(rMax,b)|filter(gMax,r)|filter(bMax,g);
const v8hi h=cA+div_s(lshift<7>(cB-cC),d);
out_h=(h+make_v8hi(256))&make_v8hi(255);
// out_s=div(mul(make_v8hi(85),d),max(make_v8hi(1),maxC)); // overflow!
out_s=div_u(mul(make_v8hi(85/2),d),max(make_v8hi(1),rshift<1>(maxC)));
out_v=maxC;
}

inline
float
scalar_hsv_sqrDist(int hTst, int sTst, int vTst,
                   int hRef, int sRef, int vRef,
                   float hWeight, float sWeight, float vWeight)
{
#if 1
  const float dh=float(std::min((256+hRef-hTst)&255,
                                (256+hTst-hRef)&255))*(1.0f/128.0f);
  const float ds=float(sRef-sTst)*(1.0f/255.0f);
  const float dv=float(vRef-vTst)*(1.0f/255.0f);
  return hWeight*dh*dh+sWeight*ds*ds+vWeight*dv*dv;
#else
  // NOTE: This was a previous attempt relying on the euclidian distance
  //       in the HSV cone but this distance is not specific to the
  //       reference colour.
  //       For example, two similar colours with different V will be
  //       considered far from each other although we know that this
  //       distance is only due to different lighting conditions.
  //
  (void)hWeight; (void)sWeight; (void)vWeight; // weights are unused here
  // Bhaskara I's sine approximation:
  //   cos(x)=(pi^2 - 4*x^2)/(pi^2 + x^2) in [-pi/2;pi/2] (error<0.001631)
  const int dh=std::min((256+hRef-hTst)&255,
                        (256+hTst-hRef)&255); // [0-128]
  const bool sym=dh>=64; // angle >= pi/2
  float angle=float(dh)*(M_PIf/128.0f);
  if(sym) { angle=M_PIf-angle; }
  const float sqrAngle=angle*angle;
  float cosAngle=(M_PIf*M_PIf-4.0f*sqrAngle)/(M_PIf*M_PIf+sqrAngle);
  if(sym) { cosAngle=-cosAngle; }
  // euclidian distance within a cone with axis s=0
  const float sRefNorm=float(sRef)*(1.0f/255.0f);
  const float vRefNorm=float(vRef)*(1.0f/255.0f);
  const float sTstNorm=float(sTst)*(1.0f/255.0f);
  const float vTstNorm=float(vTst)*(1.0f/255.0f);
  const float radiusRef=sRefNorm*vRefNorm,
              radiusTst=sTstNorm*vTstNorm;
  const float dv=vRefNorm-vTstNorm;
  return (dv*dv)+
         (radiusRef*radiusRef+radiusTst*radiusTst-
          2.0f*radiusRef*radiusTst*cosAngle);
#endif
}

inline
v4sf
simd_hsv_sqrDist(v4si hTst, v4si sTst, v4si vTst,
                 v4si hRef, v4si sRef, v4si vRef,
                 v4sf hWeight, v4sf sWeight, v4sf vWeight)
{
#if 1
  const v4si c_256=make_v4si(256),
             c_255=make_v4si(255);
  const v4sf c_div_255=make_v4sf(1.0f/255.0f);
  const v4sf dh=make_v4sf(min((c_256+hRef-hTst)&c_255,
                              (c_256+hTst-hRef)&c_255))*
                make_v4sf(1.0f/128.0f);
  const v4sf ds=make_v4sf(sRef-sTst)*c_div_255;
  const v4sf dv=make_v4sf(vRef-vTst)*c_div_255;
  return hWeight*dh*dh+sWeight*ds*ds+vWeight*dv*dv;
#else
  // NOTE: This was a previous attempt relying on the euclidian distance
  //       in the HSV cone but this distance is not specific to the
  //       reference colour.
  //       For example, two similar colours with different V will be
  //       considered far from each other although we know that this
  //       distance is only due to different lighting conditions.
  //
  (void)hWeight; (void)sWeight; (void)vWeight; // weights are unused here
  // Bhaskara I's sine approximation:
  //   cos(x)=(pi^2 - 4*x^2)/(pi^2 + x^2) in [-pi/2;pi/2] (error<0.001631)
  const v4si c_256=make_v4si(256),
             c_255=make_v4si(255);
  const v4sf sqrPi=make_v4sf(M_PIf*M_PIf);
  const v4si dh=min((c_256+hRef-hTst)&c_255,
                    (c_256+hTst-hRef)&c_255); // [0-128]
  const v16qu no_sym=cmp_lt(dh,make_v4si(64)); // angle < pi/2
  v4sf angle=make_v4sf(dh)*make_v4sf(M_PIf/128.0f);
  angle=select(no_sym,angle,make_v4sf(M_PIf)-angle);
  const v4sf sqrAngle=angle*angle;
  v4sf cosAngle=(sqrPi-make_v4sf(4.0f)*sqrAngle)/(sqrPi+sqrAngle);
  cosAngle=select(no_sym,cosAngle,-cosAngle);
  // euclidian distance within a cone with axis s=0
  const v4sf sqrNorm=make_v4sf(1.0f/(255.0f*255.0f));
  const v4sf radiusRef=make_v4sf(sRef)*make_v4sf(vRef)*sqrNorm,
             radiusTst=make_v4sf(sTst)*make_v4sf(vTst)*sqrNorm;
  const v4sf dv=make_v4sf(vRef-vTst)*make_v4sf(1.0f/255.0f);
  return (dv*dv)+
         (radiusRef*radiusRef+radiusTst*radiusTst-
          make_v4sf(2.0f)*radiusRef*radiusTst*cosAngle);
#endif
}

//----------------------------------------------------------------------------

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
                        float threshold)
{
(void)full_height; // unused actually!
assert(x+width<=full_width);
assert(y+height<=full_height);
ALIGNED_PTR( const uint8_t, srcY, y_in_aligned );
ALIGNED_PTR( const uint8_t, srcU, u_in_aligned );
ALIGNED_PTR( const uint8_t, srcV, v_in_aligned );
const int offset=y*full_width+x;
srcY+=offset;
srcU+=offset;
srcV+=offset;
const float sqrThreshold=threshold*threshold;
int pixelCount=0;
for(int yId=0;yId<height;++yId)
  {
  for(int xId=0;xId<width;++xId)
    {
    int ri,gi,bi, hi,si,vi;
    scalar_yuv_to_rgb(ri,gi,bi,srcY[xId],srcU[xId],srcV[xId]);
    scalar_rgb_to_hsv(hi,si,vi,ri,gi,bi);
    const float sqrDist=scalar_hsv_sqrDist(hi,si,vi,
                                           hRef,sRef,vRef,
                                           hWeight,sWeight,vWeight);
    if(sqrDist<=sqrThreshold)
      { ++pixelCount; }
#if 0 // FIXME: remove (ugly hack for testing purpose only)
    uint8_t *dy=(uint8_t *)srcY, *du=(uint8_t *)srcU, *dv=(uint8_t *)srcV;
    float val=192.0f*std::max(0.0f,threshold-std::sqrt(sqrDist));
    if(sqrDist<=sqrThreshold) { val+=63.0f; }
    dy[xId]=du[xId]=dv[xId]=uint8_t(val);
#endif
    }
  srcY+=full_width;
  srcU+=full_width;
  srcV+=full_width;
  }
return pixelCount;
}

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
                      float threshold)
{
(void)full_height; // unused actually!
assert(x+width<=full_width);
assert(y+height<=full_height);
assert(x%SIMD_VECTOR_SIZE==0);
assert(width%SIMD_VECTOR_SIZE==0);
assert(full_width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( const v16qu, srcY, y_in_aligned );
ALIGNED_PTR( const v16qu, srcU, u_in_aligned );
ALIGNED_PTR( const v16qu, srcV, v_in_aligned );
const int vX=x/SIMD_VECTOR_SIZE;
const int vWidth=width/SIMD_VECTOR_SIZE;
const int vFull_width=full_width/SIMD_VECTOR_SIZE;
const int offset=y*vFull_width+vX;
srcY+=offset;
srcU+=offset;
srcV+=offset;
const v4sf v_sqrThreshold=make_v4sf(threshold*threshold);
const v4si v_hRef=make_v4si(hRef),
           v_sRef=make_v4si(sRef),
           v_vRef=make_v4si(vRef);
const v4sf v_hWeight=make_v4sf(hWeight),
           v_sWeight=make_v4sf(sWeight),
           v_vWeight=make_v4sf(vWeight);
const v16qu lowMask=v16qu(make_v4si(1));
v4si pixelCount=v4si(all_zeros());
for(int yId=0;yId<height;++yId)
  {
  for(int xId=0;xId<vWidth;++xId)
    {
    const v16qu yy=load_a(srcY+xId);
    const v16qu uu=load_a(srcU+xId);
    const v16qu vv=load_a(srcV+xId);
    v4si r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3;
    v8hi h0,s0,v0, h1,s1,v1;
    simd_yuv_to_rgb(r0,g0,b0,
                    make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
    simd_yuv_to_rgb(r1,g1,b1,
                    make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
    simd_rgb_to_hsv(h0,s0,v0,
                    make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
    simd_yuv_to_rgb(r2,g2,b2,
                    make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
    simd_yuv_to_rgb(r3,g3,b3,
                    make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
    simd_rgb_to_hsv(h1,s1,v1,
                    make_v8hi(r2,r3),make_v8hi(g2,g3),make_v8hi(b2,b3));
    const v4sf sqrDist0=simd_hsv_sqrDist(
                        make_v4si_0u(h0),make_v4si_0u(s0),make_v4si_0u(v0),
                        v_hRef,v_sRef,v_vRef,v_hWeight,v_sWeight,v_vWeight),
               sqrDist1=simd_hsv_sqrDist(
                        make_v4si_1u(h0),make_v4si_1u(s0),make_v4si_1u(v0),
                        v_hRef,v_sRef,v_vRef,v_hWeight,v_sWeight,v_vWeight),
               sqrDist2=simd_hsv_sqrDist(
                        make_v4si_0u(h1),make_v4si_0u(s1),make_v4si_0u(v1),
                        v_hRef,v_sRef,v_vRef,v_hWeight,v_sWeight,v_vWeight),
               sqrDist3=simd_hsv_sqrDist(
                        make_v4si_1u(h1),make_v4si_1u(s1),make_v4si_1u(v1),
                        v_hRef,v_sRef,v_vRef,v_hWeight,v_sWeight,v_vWeight);
    pixelCount+=v4si(filter(lowMask,cmp_le(sqrDist0,v_sqrThreshold)))+
                v4si(filter(lowMask,cmp_le(sqrDist1,v_sqrThreshold)))+
                v4si(filter(lowMask,cmp_le(sqrDist2,v_sqrThreshold)))+
                v4si(filter(lowMask,cmp_le(sqrDist3,v_sqrThreshold)));
    }
  srcY+=vFull_width;
  srcU+=vFull_width;
  srcV+=vFull_width;
  }
return hsum(pixelCount);
}

//----------------------------------------------------------------------------

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
                  int full_height)
{
(void)full_height; // unused actually!
assert(x+width<=full_width);
assert(y+height<=full_height);
ALIGNED_PTR( uint8_t, dstH, h_out_aligned );
ALIGNED_PTR( uint8_t, dstS, s_out_aligned );
ALIGNED_PTR( uint8_t, dstV, v_out_aligned );
ALIGNED_PTR( const uint8_t, srcY, y_in_aligned );
ALIGNED_PTR( const uint8_t, srcU, u_in_aligned );
ALIGNED_PTR( const uint8_t, srcV, v_in_aligned );
const int offset=y*full_width+x;
srcY+=offset;
srcU+=offset;
srcV+=offset;
for(int yId=0;yId<height;++yId)
  {
  for(int xId=0;xId<width;++xId)
    {
    int ri,gi,bi, hi,si,vi;
    scalar_yuv_to_rgb(ri,gi,bi,srcY[xId],srcU[xId],srcV[xId]);
    scalar_rgb_to_hsv(hi,si,vi,ri,gi,bi);
    dstH[xId]=uint8_t(hi);
    dstS[xId]=uint8_t(si);
    dstV[xId]=uint8_t(vi);
    }
  srcY+=full_width;
  srcU+=full_width;
  srcV+=full_width;
  dstH+=width;
  dstS+=width;
  dstV+=width;
  }
}

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
                int full_height)
{
(void)full_height; // unused actually!
assert(x+width<=full_width);
assert(y+height<=full_height);
assert(x%SIMD_VECTOR_SIZE==0);
assert(width%SIMD_VECTOR_SIZE==0);
assert(full_width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstH, h_out_aligned );
ALIGNED_PTR( v16qu, dstS, s_out_aligned );
ALIGNED_PTR( v16qu, dstV, v_out_aligned );
ALIGNED_PTR( const v16qu, srcY, y_in_aligned );
ALIGNED_PTR( const v16qu, srcU, u_in_aligned );
ALIGNED_PTR( const v16qu, srcV, v_in_aligned );
const int vX=x/SIMD_VECTOR_SIZE;
const int vWidth=width/SIMD_VECTOR_SIZE;
const int vFull_width=full_width/SIMD_VECTOR_SIZE;
const int offset=y*vFull_width+vX;
srcY+=offset;
srcU+=offset;
srcV+=offset;
for(int yId=0;yId<height;++yId)
  {
  for(int xId=0;xId<vWidth;++xId)
    {
    const v16qu yy=load_a(srcY+xId);
    const v16qu uu=load_a(srcU+xId);
    const v16qu vv=load_a(srcV+xId);
    v4si r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3;
    v8hi h0,s0,v0, h1,s1,v1;
    simd_yuv_to_rgb(r0,g0,b0,
                    make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
    simd_yuv_to_rgb(r1,g1,b1,
                    make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
    simd_rgb_to_hsv(h0,s0,v0,
                    make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
    simd_yuv_to_rgb(r2,g2,b2,
                    make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
    simd_yuv_to_rgb(r3,g3,b3,
                    make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
    simd_rgb_to_hsv(h1,s1,v1,
                    make_v8hi(r2,r3),make_v8hi(g2,g3),make_v8hi(b2,b3));
    store_a(dstH+xId,make_v16qu(h0,h1));
    store_a(dstS+xId,make_v16qu(s0,s1));
    store_a(dstV+xId,make_v16qu(v0,v1));
    }
  srcY+=vFull_width;
  srcU+=vFull_width;
  srcV+=vFull_width;
  dstH+=vWidth;
  dstS+=vWidth;
  dstV+=vWidth;
  }
}

//----------------------------------------------------------------------------

void
scalar_V4L_to_YHSV(uint8_t *y_out_aligned,
                   uint8_t *h_out_aligned,
                   uint8_t *s_out_aligned,
                   uint8_t *v_out_aligned,
                   const uint8_t *v4l_in_aligned,
                   int width,
                   int height)
{
ALIGNED_PTR( uint8_t, dstY, y_out_aligned );
ALIGNED_PTR( uint8_t, dstH, h_out_aligned );
ALIGNED_PTR( uint8_t, dstS, s_out_aligned );
ALIGNED_PTR( uint8_t, dstV, v_out_aligned );
#if FULL_FRAME
  ALIGNED_PTR( const uint8_t, src, v4l_in_aligned );
  const int pixelCount=width*height;
  for(int id=0;id<pixelCount;++id)
    {
    const int y=src[2*id];
    const int u=src[((2*id)&~3)|1];
    const int v=src[((2*id)&~3)|3];
    dstY[id]=uint8_t(y);
    int ri,gi,bi, hi,si,vi;
    scalar_yuv_to_rgb(ri,gi,bi,y,u,v);
    scalar_rgb_to_hsv(hi,si,vi,ri,gi,bi);
    dstH[id]=uint8_t(hi); dstS[id]=uint8_t(si); dstV[id]=uint8_t(vi);
    }
#else
  ALIGNED_PTR( const uint8_t, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const uint8_t, src1, src0+width*4 );
    for(int xId=0;xId<width;++xId)
      {
      // emulate _mm_avg_epu8(a,b) as (a+b+1)>>1
      const int y0=(src0[4*xId+0]+src1[4*xId+0]+1)>>1;
      const int u1=(src0[4*xId+1]+src1[4*xId+1]+1)>>1;
      const int y2=(src0[4*xId+2]+src1[4*xId+2]+1)>>1;
      const int v3=(src0[4*xId+3]+src1[4*xId+3]+1)>>1;
      const int y02=(y0+y2+1)>>1;
      dstY[xId]=uint8_t(y02);
      int ri,gi,bi, hi,si,vi;
      scalar_yuv_to_rgb(ri,gi,bi,y02,u1,v3);
      scalar_rgb_to_hsv(hi,si,vi,ri,gi,bi);
      dstH[xId]=uint8_t(hi); dstS[xId]=uint8_t(si); dstV[xId]=uint8_t(vi);
      }
    src0+=width*8;
    dstY+=width;
    dstH+=width;
    dstS+=width;
    dstV+=width;
    }
#endif
}

void
simd_V4L_to_YHSV(uint8_t *y_out_aligned,
                 uint8_t *h_out_aligned,
                 uint8_t *s_out_aligned,
                 uint8_t *v_out_aligned,
                 const uint8_t *v4l_in_aligned,
                 int width,
                 int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstY, y_out_aligned );
ALIGNED_PTR( v16qu, dstH, h_out_aligned );
ALIGNED_PTR( v16qu, dstS, s_out_aligned );
ALIGNED_PTR( v16qu, dstV, v_out_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
#if FULL_FRAME
  ALIGNED_PTR( const v16qu, src, v4l_in_aligned );
  const int vPixelCount=vWidth*height;
  for(int id=0;id<vPixelCount;++id)
    {
    // extract y0y1y2y3... u0u0u1u1... v0v0v1v1... from y0u0y1v0|y2u1y3u2...
    v16qu yuyv,yy,uu,vv;
    yuyv=load_a(src+2*id+0);
    yy=get0of2_0(yuyv);
    uu=get1of4_0(yuyv);
    vv=get3of4_0(yuyv);
    yuyv=load_a(src+2*id+1);
    yy|=get0of2_1(yuyv); store_a(dstY+id,yy);
    uu|=get1of4_1(yuyv); uu=interleave_0(uu,uu);
    vv|=get3of4_1(yuyv); vv=interleave_0(vv,vv);
    // convert yyyy... uuuu... vvvv... to rrrr... gggg... bbbb...
    //                                 to hhhh... ssss... vvvv...
    v4si r0,g0,b0, r1,g1,b1;
    v8hi h0,s0,v0, h1,s1,v1;
    simd_yuv_to_rgb(r0,g0,b0,
                    make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
    simd_yuv_to_rgb(r1,g1,b1,
                    make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
    simd_rgb_to_hsv(h0,s0,v0,
                    make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
    simd_yuv_to_rgb(r0,g0,b0,
                    make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
    simd_yuv_to_rgb(r1,g1,b1,
                    make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
    simd_rgb_to_hsv(h1,s1,v1,
                    make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
    store_a(dstH+id,make_v16qu(h0,h1));
    store_a(dstS+id,make_v16qu(s0,s1));
    store_a(dstV+id,make_v16qu(v0,v1));
    }
#else
  ALIGNED_PTR( const v16qu, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const v16qu, src1, src0+vWidth*4 );
    for(int xId=0;xId<vWidth;++xId)
      {
      // extract yyyy... uuuu... vvvv... from yuyv|yuyu...
      v16qu yuv,yy,uu,vv;
      yuv=avg(load_a(src0+4*xId+0),load_a(src1+4*xId+0));
      yy=avg(get0of4_0(yuv),get2of4_0(yuv));
      uu=get1of4_0(yuv);
      vv=get3of4_0(yuv);
      yuv=avg(load_a(src0+4*xId+1),load_a(src1+4*xId+1));
      yy|=avg(get0of4_1(yuv),get2of4_1(yuv));
      uu|=get1of4_1(yuv);
      vv|=get3of4_1(yuv);
      yuv=avg(load_a(src0+4*xId+2),load_a(src1+4*xId+2));
      yy|=avg(get0of4_2(yuv),get2of4_2(yuv));
      uu|=get1of4_2(yuv);
      vv|=get3of4_2(yuv);
      yuv=avg(load_a(src0+4*xId+3),load_a(src1+4*xId+3));
      yy|=avg(get0of4_3(yuv),get2of4_3(yuv)); store_a(dstY+xId,yy);
      uu|=get1of4_3(yuv);
      vv|=get3of4_3(yuv);
      // convert yyyy... uuuu... vvvv... to rrrr... gggg... bbbb...
      //                                 to hhhh... ssss... vvvv...
      v4si r0,g0,b0, r1,g1,b1;
      v8hi h0,s0,v0, h1,s1,v1;
      simd_yuv_to_rgb(r0,g0,b0,
                      make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
      simd_yuv_to_rgb(r1,g1,b1,
                      make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
      simd_rgb_to_hsv(h0,s0,v0,
                      make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
      simd_yuv_to_rgb(r0,g0,b0,
                      make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
      simd_yuv_to_rgb(r1,g1,b1,
                      make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
      simd_rgb_to_hsv(h1,s1,v1,
                      make_v8hi(r0,r1),make_v8hi(g0,g1),make_v8hi(b0,b1));
      store_a(dstH+xId,make_v16qu(h0,h1));
      store_a(dstS+xId,make_v16qu(s0,s1));
      store_a(dstV+xId,make_v16qu(v0,v1));
      }
    src0+=vWidth*8;
    dstY+=vWidth;
    dstH+=vWidth;
    dstS+=vWidth;
    dstV+=vWidth;
    }
#endif
}

//----------------------------------------------------------------------------

void
scalar_V4L_to_YRGB(uint8_t *y_out_aligned,
                   uint8_t *r_out_aligned,
                   uint8_t *g_out_aligned,
                   uint8_t *b_out_aligned,
                   const uint8_t *v4l_in_aligned,
                   int width,
                   int height)
{
ALIGNED_PTR( uint8_t, dstY, y_out_aligned );
ALIGNED_PTR( uint8_t, dstR, r_out_aligned );
ALIGNED_PTR( uint8_t, dstG, g_out_aligned );
ALIGNED_PTR( uint8_t, dstB, b_out_aligned );
#if FULL_FRAME
  ALIGNED_PTR( const uint8_t, src, v4l_in_aligned );
  const int pixelCount=width*height;
  for(int id=0;id<pixelCount;++id)
    {
    const int y=src[2*id];
    const int u=src[((2*id)&~3)|1];
    const int v=src[((2*id)&~3)|3];
    dstY[id]=uint8_t(y);
    int ri,gi,bi;
    scalar_yuv_to_rgb(ri,gi,bi,y,u,v);
    dstR[id]=uint8_t(ri); dstG[id]=uint8_t(gi); dstB[id]=uint8_t(bi);
    }
#else
  ALIGNED_PTR( const uint8_t, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const uint8_t, src1, src0+width*4 );
    for(int xId=0;xId<width;++xId)
      {
      // emulate _mm_avg_epu8(a,b) as (a+b+1)>>1
      const int y0=(src0[4*xId+0]+src1[4*xId+0]+1)>>1;
      const int u1=(src0[4*xId+1]+src1[4*xId+1]+1)>>1;
      const int y2=(src0[4*xId+2]+src1[4*xId+2]+1)>>1;
      const int v3=(src0[4*xId+3]+src1[4*xId+3]+1)>>1;
      const int y02=(y0+y2+1)>>1;
      dstY[xId]=uint8_t(y02);
      int ri,gi,bi;
      scalar_yuv_to_rgb(ri,gi,bi,y02,u1,v3);
      dstR[xId]=uint8_t(ri); dstG[xId]=uint8_t(gi); dstB[xId]=uint8_t(bi);
      }
    src0+=width*8;
    dstY+=width;
    dstR+=width;
    dstG+=width;
    dstB+=width;
    }
#endif
}

void
simd_V4L_to_YRGB(uint8_t *y_out_aligned,
                 uint8_t *r_out_aligned,
                 uint8_t *g_out_aligned,
                 uint8_t *b_out_aligned,
                 const uint8_t *v4l_in_aligned,
                 int width,
                 int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstY, y_out_aligned );
ALIGNED_PTR( v16qu, dstR, r_out_aligned );
ALIGNED_PTR( v16qu, dstG, g_out_aligned );
ALIGNED_PTR( v16qu, dstB, b_out_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
#if FULL_FRAME
  ALIGNED_PTR( const v16qu, src, v4l_in_aligned );
  const int vPixelCount=vWidth*height;
  for(int id=0;id<vPixelCount;++id)
    {
    // extract y0y1y2y3... u0u0u1u1... v0v0v1v1... from y0u0y1v0|y2u1y3u2...
    v16qu yuyv,yy,uu,vv;
    yuyv=load_a(src+2*id+0);
    yy=get0of2_0(yuyv);
    uu=get1of4_0(yuyv);
    vv=get3of4_0(yuyv);
    yuyv=load_a(src+2*id+1);
    yy|=get0of2_1(yuyv); store_a(dstY+id,yy);
    uu|=get1of4_1(yuyv); uu=interleave_0(uu,uu);
    vv|=get3of4_1(yuyv); vv=interleave_0(vv,vv);
    // convert yyyy... uuuu... vvvv... to rrrr... gggg... bbbb...
    v4si r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3;
    simd_yuv_to_rgb(r0,g0,b0,
                    make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
    simd_yuv_to_rgb(r1,g1,b1,
                    make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
    simd_yuv_to_rgb(r2,g2,b2,
                    make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
    simd_yuv_to_rgb(r3,g3,b3,
                    make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
    store_a(dstR+id,make_v16qu(r0,r1,r2,r3));
    store_a(dstG+id,make_v16qu(g0,g1,g2,g3));
    store_a(dstB+id,make_v16qu(b0,b1,b2,b3));
    }
#else
  ALIGNED_PTR( const v16qu, src0, v4l_in_aligned );
  for(int yId=0;yId<height;++yId)
    {
    ALIGNED_PTR( const v16qu, src1, src0+vWidth*4 );
    for(int xId=0;xId<vWidth;++xId)
      {
      // extract yyyy... uuuu... vvvv... from yuyv|yuyu...
      v16qu yuv,yy,uu,vv;
      yuv=avg(load_a(src0+4*xId+0),load_a(src1+4*xId+0));
      yy=avg(get0of4_0(yuv),get2of4_0(yuv));
      uu=get1of4_0(yuv);
      vv=get3of4_0(yuv);
      yuv=avg(load_a(src0+4*xId+1),load_a(src1+4*xId+1));
      yy|=avg(get0of4_1(yuv),get2of4_1(yuv));
      uu|=get1of4_1(yuv);
      vv|=get3of4_1(yuv);
      yuv=avg(load_a(src0+4*xId+2),load_a(src1+4*xId+2));
      yy|=avg(get0of4_2(yuv),get2of4_2(yuv));
      uu|=get1of4_2(yuv);
      vv|=get3of4_2(yuv);
      yuv=avg(load_a(src0+4*xId+3),load_a(src1+4*xId+3));
      yy|=avg(get0of4_3(yuv),get2of4_3(yuv)); store_a(dstY+xId,yy);
      uu|=get1of4_3(yuv);
      vv|=get3of4_3(yuv);
      // convert yyyy... uuuu... vvvv... to rrrr... gggg... bbbb...
      v4si r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3;
      simd_yuv_to_rgb(r0,g0,b0,
                      make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
      simd_yuv_to_rgb(r1,g1,b1,
                      make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
      simd_yuv_to_rgb(r2,g2,b2,
                      make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
      simd_yuv_to_rgb(r3,g3,b3,
                      make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
      store_a(dstR+xId,make_v16qu(r0,r1,r2,r3));
      store_a(dstG+xId,make_v16qu(g0,g1,g2,g3));
      store_a(dstB+xId,make_v16qu(b0,b1,b2,b3));
      }
    src0+=vWidth*8;
    dstY+=vWidth;
    dstR+=vWidth;
    dstG+=vWidth;
    dstB+=vWidth;
    }
#endif
}

//----------------------------------------------------------------------------

void
scalar_YUV_to_RGB(uint8_t *r_out_aligned,
                  uint8_t *g_out_aligned,
                  uint8_t *b_out_aligned,
                  const uint8_t *y_in_aligned,
                  const uint8_t *u_in_aligned,
                  const uint8_t *v_in_aligned,
                  int width,
                  int height)
{
ALIGNED_PTR( uint8_t, dstR, r_out_aligned );
ALIGNED_PTR( uint8_t, dstG, g_out_aligned );
ALIGNED_PTR( uint8_t, dstB, b_out_aligned );
ALIGNED_PTR( const uint8_t, srcY, y_in_aligned );
ALIGNED_PTR( const uint8_t, srcU, u_in_aligned );
ALIGNED_PTR( const uint8_t, srcV, v_in_aligned );
const int pixelCount=width*height;
for(int id=0;id<pixelCount;++id)
  {
  int ri,gi,bi;
  scalar_yuv_to_rgb(ri,gi,bi,srcY[id],srcU[id],srcV[id]);
  dstR[id]=uint8_t(ri); dstG[id]=uint8_t(gi); dstB[id]=uint8_t(bi);
  }
}

void
simd_YUV_to_RGB(uint8_t *r_out_aligned,
                uint8_t *g_out_aligned,
                uint8_t *b_out_aligned,
                const uint8_t *y_in_aligned,
                const uint8_t *u_in_aligned,
                const uint8_t *v_in_aligned,
                int width,
                int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstR, r_out_aligned );
ALIGNED_PTR( v16qu, dstG, g_out_aligned );
ALIGNED_PTR( v16qu, dstB, b_out_aligned );
ALIGNED_PTR( const v16qu, srcY, y_in_aligned );
ALIGNED_PTR( const v16qu, srcU, u_in_aligned );
ALIGNED_PTR( const v16qu, srcV, v_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
const int vPixelCount=vWidth*height;
for(int id=0;id<vPixelCount;++id)
  {
  const v16qu yy=load_a(srcY+id);
  const v16qu uu=load_a(srcU+id);
  const v16qu vv=load_a(srcV+id);
  v4si r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3;
  simd_yuv_to_rgb(r0,g0,b0,
                  make_v4si_0u(yy),make_v4si_0u(uu),make_v4si_0u(vv));
  simd_yuv_to_rgb(r1,g1,b1,
                  make_v4si_1u(yy),make_v4si_1u(uu),make_v4si_1u(vv));
  simd_yuv_to_rgb(r2,g2,b2,
                  make_v4si_2u(yy),make_v4si_2u(uu),make_v4si_2u(vv));
  simd_yuv_to_rgb(r3,g3,b3,
                  make_v4si_3u(yy),make_v4si_3u(uu),make_v4si_3u(vv));
  store_a(dstR+id,make_v16qu(r0,r1,r2,r3));
  store_a(dstG+id,make_v16qu(g0,g1,g2,g3));
  store_a(dstB+id,make_v16qu(b0,b1,b2,b3));
  }
}

//----------------------------------------------------------------------------

void
scalar_RGB_to_HSV(uint8_t *h_out_aligned,
                  uint8_t *s_out_aligned,
                  uint8_t *v_out_aligned,
                  const uint8_t *r_in_aligned,
                  const uint8_t *g_in_aligned,
                  const uint8_t *b_in_aligned,
                  int width,
                  int height)
{
ALIGNED_PTR( uint8_t, dstH, h_out_aligned );
ALIGNED_PTR( uint8_t, dstS, s_out_aligned );
ALIGNED_PTR( uint8_t, dstV, v_out_aligned );
ALIGNED_PTR( const uint8_t, srcR, r_in_aligned );
ALIGNED_PTR( const uint8_t, srcG, g_in_aligned );
ALIGNED_PTR( const uint8_t, srcB, b_in_aligned );
const int pixelCount=width*height;
for(int id=0;id<pixelCount;++id)
  {
  int hi,si,vi;
  scalar_rgb_to_hsv(hi,si,vi,srcR[id],srcG[id],srcB[id]);
  dstH[id]=uint8_t(hi); dstS[id]=uint8_t(si); dstV[id]=uint8_t(vi);
  }
}

void
simd_RGB_to_HSV(uint8_t *h_out_aligned,
                uint8_t *s_out_aligned,
                uint8_t *v_out_aligned,
                const uint8_t *r_in_aligned,
                const uint8_t *g_in_aligned,
                const uint8_t *b_in_aligned,
                int width,
                int height)
{
assert(width%SIMD_VECTOR_SIZE==0);
enterSimd();
ALIGNED_PTR( v16qu, dstH, h_out_aligned );
ALIGNED_PTR( v16qu, dstS, s_out_aligned );
ALIGNED_PTR( v16qu, dstV, v_out_aligned );
ALIGNED_PTR( const v16qu, srcR, r_in_aligned );
ALIGNED_PTR( const v16qu, srcG, g_in_aligned );
ALIGNED_PTR( const v16qu, srcB, b_in_aligned );
const int vWidth=width/SIMD_VECTOR_SIZE;
const int vPixelCount=vWidth*height;
for(int id=0;id<vPixelCount;++id)
  {
  const v16qu rr=load_a(srcR+id);
  const v16qu gg=load_a(srcG+id);
  const v16qu bb=load_a(srcB+id);
  v8hi h0,s0,v0, h1,s1,v1;
  simd_rgb_to_hsv(h0,s0,v0,
                  make_v8hi_0u(rr),make_v8hi_0u(gg),make_v8hi_0u(bb));
  simd_rgb_to_hsv(h1,s1,v1,
                  make_v8hi_1u(rr),make_v8hi_1u(gg),make_v8hi_1u(bb));
  store_a(dstH+id,make_v16qu(h0,h1));
  store_a(dstS+id,make_v16qu(s0,s1));
  store_a(dstV+id,make_v16qu(v0,v1));
  }
}

//----------------------------------------------------------------------------

FramePyramid::FramePyramid()
: m_xPyramid()
, m_yPyramid()
{
}

FramePyramid::FramePyramid(int minSizeX,
                           int fullSizeX,
                           float scaleFactorX,
                           int slideX,
                           int minSizeY,
                           int fullSizeY,
                           float scaleFactorY,
                           int slideY)
: m_xPyramid()
, m_yPyramid()
{
m_init(m_xPyramid,minSizeX,fullSizeX,scaleFactorX,slideX);
m_init(m_yPyramid,minSizeY,fullSizeY,scaleFactorY,slideY);
while(m_xPyramid.size()>m_yPyramid.size())
  { m_xPyramid.pop_back(); }
while(m_yPyramid.size()>m_xPyramid.size())
  { m_yPyramid.pop_back(); }
}

void
FramePyramid::m_init(std::vector<SizePos> &pyramid,
                     int minSize,
                     int fullSize,
                     float scaleFactor,
                     int slide)
{
assert(fullSize>=minSize);
assert(scaleFactor>1.0f);
float fSize=float(minSize);
SizePos elem;
for(elem.size=minSize;
    elem.size<=fullSize;
    fSize*=scaleFactor, elem.size=int(std::round(fSize)))
  {
  if(pyramid.empty()||(pyramid.back().size!=elem.size))
    { pyramid.push_back(elem); }
  }
for(int i=0;i<int(pyramid.size());++i)
  {
  std::vector<int> &vPos=pyramid[i].pos;
  const int size=pyramid[i].size;
  float fStep=float(size)/float(slide);
  const int innerSize=fullSize-size;
  const float fStepCount=float(innerSize)/fStep;
  const int stepCount=int(std::round(fStepCount));
  float fPos=0.0f;
  if(stepCount==0)
    { fPos=0.5f*float(fullSize-size); }
  else if(stepCount==1)
    { fStep=float(fullSize-size); }
  else
    { fStep=float(innerSize)/float(stepCount); }
  for(int i=0;i<=stepCount;++i, fPos+=fStep)
    {
    const int pos=int(std::round(fPos));
    if(vPos.empty()||(vPos.back()!=pos))
      { vPos.push_back(pos); }
    }
  }
}

//----------------------------------------------------------------------------

FrameTimer::FrameTimer(std::string title)
: m_title(title)
, m_accum(0.0)
, m_last(now())
, m_count(0)
{ }

void
FrameTimer::tic(double startTime)
{
const double endTime=now();
m_accum+=endTime-startTime;
++m_count;
double dt=endTime-m_last;
if(dt>=2.0)
  {
  const double fps=m_count/dt;
  const double ms=1e3*m_accum/m_count;
  std::cerr << m_title << ": " << fps << " fps, " << ms <<  " ms/frame\n";
  m_accum=0.0;
  m_last=endTime;
  m_count=0;
  }
}

double // seconds since 1970/01/01 00:00:00 UTC
FrameTimer::now()
{
struct timeval tv;
gettimeofday(&tv,NULL);
return double(tv.tv_sec)+1e-6*double(tv.tv_usec);
}

//----------------------------------------------------------------------------
