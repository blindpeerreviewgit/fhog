#ifndef HOGBACKEND_HPP
#define HOGBACKEND_HPP 1

#include "frameUtils.hpp"
#include <cmath>

class HogBackend
{
public:

  HogBackend()
  : m_yuvStorage    (storage(3*pixelCount))
  , m_rgbStorage    (storage(3*pixelCount))
  , m_hsvStorage    (storage(3*pixelCount))
  , m_dxyStorage    (storage(2*pixelCount))
  , m_modArgStorage (storage(2*pixelCount))
  , m_histoStorage  (storage<float>(histoValueCount))
  , m_extractStorage(storage<float>(extractValueCount))
  , m_extracttmpStorage(storage<float>(extractValueCount))
  , m_equalStorage  (storage<float>(equalValueCount))
  , m_yuv           (aligned(m_yuvStorage.get()))
  , m_rgb           (aligned(m_rgbStorage.get()))
  , m_hsv           (aligned(m_hsvStorage.get()))
  , m_dxy           (aligned(m_dxyStorage.get()))
  , m_modArg        (aligned(m_modArgStorage.get()))
  , m_histo         (aligned(m_histoStorage.get()))
  , m_extract       (aligned(m_extractStorage.get()))
  , m_extracttmp       (aligned(m_extracttmpStorage.get()))
  , m_equal         (aligned(m_equalStorage.get()))
  , m_fp            ()
  {
  // these parameters may be freely chosen
  const float scaleFactor=std::pow(2.0f,1.0f/4.0f); // x2 every 4 iterations
  const int slide=3; // slide sub-region from 1/3 of its size
  m_fp=FramePyramid(extractWidth,histoWidth,scaleFactor,slide,
                    extractHeight,histoHeight,scaleFactor,slide);
  }

  enum
  {
  width             =FRAME_WIDTH,
  height            =FRAME_HEIGHT,
  pixelCount        =width*height,

  histoWidth        =width/HISTO_CELL_SIZE,
  histoHeight       =height/HISTO_CELL_SIZE,
  histoValueCount   =histoWidth*histoHeight*HISTO_BIN_COUNT,

  extractWidth      =12, // 80 pixels wide
  extractHeight     =16, // 80 pixels high
  extractValueCount =histoValueCount,//extractWidth*extractHeight*HISTO_BIN_COUNT,

  equalBlock        =2, // 2x2 cells block
  equalWidth        =extractWidth+1-equalBlock,
  equalHeight       =extractHeight+1-equalBlock,
  equalValueCount   =(equalWidth*equalBlock)*(equalHeight*equalBlock)*
                     HISTO_BIN_COUNT
  };

        uint8_t * y()             { return m_yuv+0*pixelCount;    }
  const uint8_t * y()       const { return m_yuv+0*pixelCount;    }

        uint8_t * u()             { return m_yuv+1*pixelCount;    }
  const uint8_t * u()       const { return m_yuv+1*pixelCount;    }

        uint8_t * v()             { return m_yuv+2*pixelCount;    }
  const uint8_t * v()       const { return m_yuv+2*pixelCount;    }

        uint8_t * red()           { return m_rgb+0*pixelCount;    }
  const uint8_t * red()     const { return m_rgb+0*pixelCount;    }

        uint8_t * green()         { return m_rgb+1*pixelCount;    }
  const uint8_t * green()   const { return m_rgb+1*pixelCount;    }

        uint8_t * blue()          { return m_rgb+2*pixelCount;    }
  const uint8_t * blue()    const { return m_rgb+2*pixelCount;    }

        uint8_t * hue()           { return m_hsv+0*pixelCount;    }
  const uint8_t * hue()     const { return m_hsv+0*pixelCount;    }

        uint8_t * sat()           { return m_hsv+1*pixelCount;    }
  const uint8_t * sat()     const { return m_hsv+1*pixelCount;    }

        uint8_t * val()           { return m_hsv+2*pixelCount;    }
  const uint8_t * val()     const { return m_hsv+2*pixelCount;    }

        uint8_t * dx()            { return m_dxy+0*pixelCount;    }
  const uint8_t * dx()      const { return m_dxy+0*pixelCount;    }

        uint8_t * dy()            { return m_dxy+1*pixelCount;    }
  const uint8_t * dy()      const { return m_dxy+1*pixelCount;    }

        uint8_t * mod()           { return m_modArg+0*pixelCount; }
  const uint8_t * mod()     const { return m_modArg+0*pixelCount; }

        uint8_t * arg()           { return m_modArg+1*pixelCount; }
  const uint8_t * arg()     const { return m_modArg+1*pixelCount; }

        float   * histo()         { return m_histo;               }
  const float   * histo()   const { return m_histo;               }

        float   * extract()       { return m_extract;             }
  const float   * extract() const { return m_extract;             }

        float   * extracttmp()       { return m_extracttmp;             }
  const float   * extracttmp() const { return m_extracttmp;             }

        void swapExtract() {float *tmp = m_extract; m_extract = m_extracttmp; m_extracttmp = tmp;}

        float   * equal()         { return m_equal;               }
  const float   * equal()   const { return m_equal;               }

  const FramePyramid & pyramid() const { return m_fp; }

private:
  std::unique_ptr<uint8_t[]> m_yuvStorage,
                             m_rgbStorage,
                             m_hsvStorage,
                             m_dxyStorage,
                             m_modArgStorage;
  std::unique_ptr<float[]> m_histoStorage,
                           m_extractStorage,
                           m_extracttmpStorage,
                           m_equalStorage;
  uint8_t *m_yuv, *m_rgb, *m_hsv, *m_dxy, *m_modArg;
  float *m_histo, *m_extract, *m_extracttmp, *m_equal;
  FramePyramid m_fp;
};

#endif // HOGBACKEND_HPP

//---------------------------------------------------------------------------
