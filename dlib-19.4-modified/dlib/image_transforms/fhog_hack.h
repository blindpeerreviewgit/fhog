// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_fHOG_Hack_
#define DLIB_fHOG_Hack_

#define DEBUG

#include <dlib/image_transforms/fhog_abstract.h>
#include <dlib/matrix.h>
#include <dlib/array2d.h>
#include <dlib/array.h>
#include <dlib/geometry.h>
#include <dlib/image_transforms/assign_image.h>
#include <dlib/image_transforms/draw.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/simd.h>

#include "hogBackend.hpp"

namespace dlib
{

    namespace impl_fhog
    {

    template <
        typename image_type, 
        typename out_type
    >
    void img_to_histo_dlib(const image_type& img_, 
                                out_type& hist, 
                                int cell_size,
                                int filter_rows_padding,
                                int filter_cols_padding)
    {
    /*
            This function implements the HOG feature extraction method described in 
            the paper:
                P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
                Object Detection with Discriminatively Trained Part Based Models
                IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010

            Moreover, this function is derived from the HOG feature extraction code
            from the features.cc file in the voc-releaseX code (see
            http://people.cs.uchicago.edu/~rbg/latent/) which is has the following
            license (note that the code has been modified to work with grayscale and
            color as well as planar and interlaced input and output formats):

            Copyright (C) 2011, 2012 Ross Girshick, Pedro Felzenszwalb
            Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
            Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan

            Permission is hereby granted, free of charge, to any person obtaining
            a copy of this software and associated documentation files (the
            "Software"), to deal in the Software without restriction, including
            without limitation the rights to use, copy, modify, merge, publish,
            distribute, sublicense, and/or sell copies of the Software, and to
            permit persons to whom the Software is furnished to do so, subject to
            the following conditions:

            The above copyright notice and this permission notice shall be
            included in all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
            EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
            MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
            NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
            LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
            OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
            WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        */

        const_image_view<image_type> img(img_);

        // if (cell_size == 1)
        // {
        //     impl_extract_fhog_features_hack_cell_size_1(fabHog,hog,filter_rows_padding,filter_cols_padding);
        //     return;
        // }

        // unit vectors used to compute gradient orientation
        matrix<float,2,1> directions[9];
        directions[0] =  1.0000, 0.0000; 
        directions[1] =  0.9397, 0.3420;
        directions[2] =  0.7660, 0.6428;
        directions[3] =  0.500,  0.8660;
        directions[4] =  0.1736, 0.9848;
        directions[5] = -0.1736, 0.9848;
        directions[6] = -0.5000, 0.8660;
        directions[7] = -0.7660, 0.6428;
        directions[8] = -0.9397, 0.3420;



        // First we allocate memory for caching orientation histograms & their norms.
        const int cells_nr = (int)((float)img.nr()/(float)cell_size + 0.5);
        const int cells_nc = (int)((float)img.nc()/(float)cell_size + 0.5);

        // std::cerr << ">>> cells_nr = " << cells_nr << "\n";
        // std::cerr << ">>> cells_nc = " << cells_nc << "\n";

        

        // We give hist extra padding around the edges (1 cell all the way around the
        // edge) so we can avoid needing to do boundary checks when indexing into it
        // later on.  So some statements assign to the boundary but those values are
        // never used.
        
        for (long r = 0; r < hist.nr(); ++r)
        {
            for (long c = 0; c < hist.nc(); ++c)
            {
                hist[r][c] = 0;
            }
        }

        const int padding_rows_offset = (filter_rows_padding-1)/2;
        const int padding_cols_offset = (filter_cols_padding-1)/2;
        
        const int visible_nr = std::min((long)cells_nr*cell_size,img.nr())-1;
        const int visible_nc = std::min((long)cells_nc*cell_size,img.nc())-1;

        // First populate the gradient histograms
        for (int y = 1; y < visible_nr; y++) 
        {
            const float yp = ((float)y+0.5)/(float)cell_size - 0.5;
            const int iyp = (int)std::floor(yp);
            const float vy0 = yp - iyp;
            const float vy1 = 1.0 - vy0;
            int x;
            for (x = 1; x < visible_nc - 7; x += 8)
            {
                simd8f xx(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                // v will be the length of the gradient vectors.
                simd8f grad_x, grad_y, v;
                get_gradient(y, x, img, grad_x, grad_y, v);

                // We will use bilinear interpolation to add into the histogram bins.
                // So first we precompute the values needed to determine how much each
                // pixel votes into each bin.
                simd8f xp = (xx + 0.5) / (float)cell_size + 0.5;
                simd8i ixp = simd8i(xp);
                simd8f vx0 = xp - ixp;
                simd8f vx1 = 1.0f - vx0;

                v = sqrt(v);

                // Now snap the gradient to one of 18 orientations
                simd8f best_dot = 0;
                simd8f best_o = 0;
                for (int o = 0; o < 9; o++)
                {
                    simd8f dot = grad_x*directions[o](0) + grad_y*directions[o](1);
                    simd8f_bool cmp = dot>best_dot;
                    best_dot = select(cmp, dot, best_dot);
                    dot *= -1;
                    best_o = select(cmp, o, best_o);

                    cmp = dot > best_dot;
                    best_dot = select(cmp, dot, best_dot);
                    best_o = select(cmp, o + 9, best_o);
                }


                // Add the gradient magnitude, v, to 4 histograms around pixel using
                // bilinear interpolation.
                vx1 *= v;
                vx0 *= v;
                // The amounts for each bin
                simd8f v11 = vy1*vx1;
                simd8f v01 = vy0*vx1;
                simd8f v10 = vy1*vx0;
                simd8f v00 = vy0*vx0;

                int32 _best_o[8]; simd8i(best_o).store(_best_o);
                int32 _ixp[8];    ixp.store(_ixp);
                float _v11[8];    v11.store(_v11);
                float _v01[8];    v01.store(_v01);
                float _v10[8];    v10.store(_v10);
                float _v00[8];    v00.store(_v00);

                hist[iyp + 1][_ixp[0]](_best_o[0]) += _v11[0];
                hist[iyp + 1 + 1][_ixp[0]](_best_o[0]) += _v01[0];
                hist[iyp + 1][_ixp[0] + 1](_best_o[0]) += _v10[0];
                hist[iyp + 1 + 1][_ixp[0] + 1](_best_o[0]) += _v00[0];

                hist[iyp + 1][_ixp[1]](_best_o[1]) += _v11[1];
                hist[iyp + 1 + 1][_ixp[1]](_best_o[1]) += _v01[1];
                hist[iyp + 1][_ixp[1] + 1](_best_o[1]) += _v10[1];
                hist[iyp + 1 + 1][_ixp[1] + 1](_best_o[1]) += _v00[1];

                hist[iyp + 1][_ixp[2]](_best_o[2]) += _v11[2];
                hist[iyp + 1 + 1][_ixp[2]](_best_o[2]) += _v01[2];
                hist[iyp + 1][_ixp[2] + 1](_best_o[2]) += _v10[2];
                hist[iyp + 1 + 1][_ixp[2] + 1](_best_o[2]) += _v00[2];

                hist[iyp + 1][_ixp[3]](_best_o[3]) += _v11[3];
                hist[iyp + 1 + 1][_ixp[3]](_best_o[3]) += _v01[3];
                hist[iyp + 1][_ixp[3] + 1](_best_o[3]) += _v10[3];
                hist[iyp + 1 + 1][_ixp[3] + 1](_best_o[3]) += _v00[3];

                hist[iyp + 1][_ixp[4]](_best_o[4]) += _v11[4];
                hist[iyp + 1 + 1][_ixp[4]](_best_o[4]) += _v01[4];
                hist[iyp + 1][_ixp[4] + 1](_best_o[4]) += _v10[4];
                hist[iyp + 1 + 1][_ixp[4] + 1](_best_o[4]) += _v00[4];

                hist[iyp + 1][_ixp[5]](_best_o[5]) += _v11[5];
                hist[iyp + 1 + 1][_ixp[5]](_best_o[5]) += _v01[5];
                hist[iyp + 1][_ixp[5] + 1](_best_o[5]) += _v10[5];
                hist[iyp + 1 + 1][_ixp[5] + 1](_best_o[5]) += _v00[5];

                hist[iyp + 1][_ixp[6]](_best_o[6]) += _v11[6];
                hist[iyp + 1 + 1][_ixp[6]](_best_o[6]) += _v01[6];
                hist[iyp + 1][_ixp[6] + 1](_best_o[6]) += _v10[6];
                hist[iyp + 1 + 1][_ixp[6] + 1](_best_o[6]) += _v00[6];

                hist[iyp + 1][_ixp[7]](_best_o[7]) += _v11[7];
                hist[iyp + 1 + 1][_ixp[7]](_best_o[7]) += _v01[7];
                hist[iyp + 1][_ixp[7] + 1](_best_o[7]) += _v10[7];
                hist[iyp + 1 + 1][_ixp[7] + 1](_best_o[7]) += _v00[7];
            }
            // Now process the right columns that don't fit into simd registers.
            for (; x < visible_nc; x++) 
            {
                matrix<float, 2, 1> grad;
                float v;
                get_gradient(y,x,img,grad,v);

                // snap to one of 18 orientations
                float best_dot = 0;
                int best_o = 0;
                for (int o = 0; o < 9; o++) 
                {
                    const float dot = dlib::dot(directions[o], grad);
                    if (dot > best_dot) 
                    {
                        best_dot = dot;
                        best_o = o;
                    } 
                    else if (-dot > best_dot) 
                    {
                        best_dot = -dot;
                        best_o = o+9;
                    }
                }

                v = std::sqrt(v);
                // add to 4 histograms around pixel using bilinear interpolation
                const float xp = ((double)x + 0.5) / (double)cell_size - 0.5;
                const int ixp = (int)std::floor(xp);
                const float vx0 = xp - ixp;
                const float vx1 = 1.0 - vx0;

                hist[iyp+1][ixp+1](best_o) += vy1*vx1*v;
                hist[iyp+1+1][ixp+1](best_o) += vy0*vx1*v;
                hist[iyp+1][ixp+1+1](best_o) += vy1*vx0*v;
                hist[iyp+1+1][ixp+1+1](best_o) += vy0*vx0*v;
            }
        }
    }

    template <
        typename T
    >
    void crop_hist(const T& hist, T& crop, const int x, const int y, const int xSize, const int ySize)
    {
        for (int i=0; i<ySize; ++i)
        {
            for (int j=0; j<xSize; ++j)
            {
                for (int b=0; b<18; ++b)
                {
                    crop[i+1][j+1](b) = hist[y+i+1][x+j+1](b);
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

        //typename image_type, 
        template <
            typename in_type,
            typename out_type
            >
        void impl_extract_fhog_features_hack2(
            const in_type& hist, 
            out_type& hog, 
            const int cells_nr,
            const int cells_nc,
            int filter_rows_padding,
            int filter_cols_padding
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT( filter_rows_padding > 0 &&
                         filter_cols_padding > 0 ,
                "\t void extract_fhog_features()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t filter_rows_padding: " << filter_rows_padding 
                << "\n\t filter_cols_padding: " << filter_cols_padding 
                );

            const int hog_nr = std::max(cells_nr-2, 0);
            const int hog_nc = std::max(cells_nc-2, 0);

            // const int padding_rows_offset = 0;
            // const int padding_cols_offset = 0;

            const int padding_rows_offset = (filter_rows_padding-1)/2;
            const int padding_cols_offset = (filter_cols_padding-1)/2;
            init_hog_zero_everything(hog, hog_nr, hog_nc, filter_rows_padding, filter_cols_padding);
            //init_hog(hog, hog_nr, hog_nc, filter_rows_padding, filter_cols_padding);

            //init_hog(hog, hog_nr, hog_nc, filter_rows_padding, filter_cols_padding);

            // std::cerr << " >>>>> " << cells_nr << " " << cells_nc << "\n";

            array2d<float> norm(cells_nr, cells_nc);
            assign_all_pixels(norm, 0);




            // std::cerr << "HISTO_CELL_SIZE=" << HISTO_CELL_SIZE << "\n";
            // std::cerr << "HISTO_BIN_WIDTH_SHIFT=" << HISTO_BIN_WIDTH_SHIFT << "\n";
            // std::cerr << "HISTO_BIN_WIDTH=" << HISTO_BIN_WIDTH << "\n";
            // std::cerr << "HISTO_BIN_WIDTH_MASK=" << HISTO_BIN_WIDTH_MASK << "\n";
            // std::cerr << "HISTO_BIN_COUNT_SHIFT=" << HISTO_BIN_COUNT_SHIFT << "\n";
            // std::cerr << "HISTO_BIN_COUNT=" << HISTO_BIN_COUNT << "\n";

            // // std::cerr << "visible_nr=" << visible_nr << "\n";
            // // std::cerr << "visible_nc=" << visible_nc << "\n";
            // std::cerr << "cells_nr=" << cells_nr << "\n";
            // std::cerr << "cells_nc=" << cells_nc << "\n";
            // //std::cerr << "cell_size=" << cell_size << "\n";
            // std::cerr << "hog_nr=" << hog_nr << "\n";
            // std::cerr << "hog_nc=" << hog_nc << "\n";

            // compute energy in each block by summing over orientations
            for (int r = 0; r < cells_nr; ++r)
            {
                for (int c = 0; c < cells_nc; ++c)
                {
                    for (int o = 0; o < 9; o++) 
                    {
                        //std::cerr << "r=" << r << " c=" << c << " o=" << o << "\n";
                        norm[r][c] += (hist[r+1][c+1](o) + hist[r+1][c+1](o+9)) * (hist[r+1][c+1](o) + hist[r+1][c+1](o+9));
                    }
                }
            }

            // std::cerr << "hi" << "\n";

            const float eps = 0.0001;
            // compute features
            for (int y = 0; y < hog_nr; y++) 
            {
                const int yy = y+padding_rows_offset; 
                for (int x = 0; x < hog_nc; x++) 
                {
                    const simd4f z1(norm[y+1][x+1],
                                    norm[y][x+1], 
                                    norm[y+1][x],  
                                    norm[y][x]);

                    const simd4f z2(norm[y+1][x+2],
                                    norm[y][x+2],
                                    norm[y+1][x+1],
                                    norm[y][x+1]);

                    const simd4f z3(norm[y+2][x+1],
                                    norm[y+1][x+1],
                                    norm[y+2][x],
                                    norm[y+1][x]);

                    const simd4f z4(norm[y+2][x+2],
                                    norm[y+1][x+2],
                                    norm[y+2][x+1],
                                    norm[y+1][x+1]);

                    const simd4f nn = 0.2*sqrt(z1+z2+z3+z4+eps);
                    const simd4f n = 0.1/nn;

                    simd4f t = 0;

                    const int xx = x+padding_cols_offset; 

                    // contrast-sensitive features
                    for (int o = 0; o < 18; o+=3) 
                    {
                        simd4f temp0(hist[y+1+1][x+1+1](o));
                        simd4f temp1(hist[y+1+1][x+1+1](o+1));
                        simd4f temp2(hist[y+1+1][x+1+1](o+2));
                        simd4f h0 = min(temp0,nn)*n;
                        simd4f h1 = min(temp1,nn)*n;
                        simd4f h2 = min(temp2,nn)*n;
                        set_hog(hog,o,xx,yy,   sum(h0));
                        set_hog(hog,o+1,xx,yy, sum(h1));
                        set_hog(hog,o+2,xx,yy, sum(h2));
                        t += h0+h1+h2;
                    }

                    t *= 2*0.2357;

                    // contrast-insensitive features
                    for (int o = 0; o < 9; o+=3) 
                    {
                        simd4f temp0 = hist[y+1+1][x+1+1](o)   + hist[y+1+1][x+1+1](o+9);
                        simd4f temp1 = hist[y+1+1][x+1+1](o+1) + hist[y+1+1][x+1+1](o+9+1);
                        simd4f temp2 = hist[y+1+1][x+1+1](o+2) + hist[y+1+1][x+1+1](o+9+2);
                        simd4f h0 = min(temp0,nn)*n;
                        simd4f h1 = min(temp1,nn)*n;
                        simd4f h2 = min(temp2,nn)*n;
                        set_hog(hog,o+18,xx,yy, sum(h0));
                        set_hog(hog,o+18+1,xx,yy, sum(h1));
                        set_hog(hog,o+18+2,xx,yy, sum(h2));
                    }


                    float temp[4];
                    t.store(temp);

                    // texture features
                    set_hog(hog,27,xx,yy, temp[0]);
                    set_hog(hog,28,xx,yy, temp[1]);
                    set_hog(hog,29,xx,yy, temp[2]);
                    set_hog(hog,30,xx,yy, temp[3]);
                }
            }

            // std::cerr << "END impl_extract_fhog_features_hack2" << "\n";
        }

        //typename image_type, 
        template <
            typename out_type
            >
        void impl_extract_fhog_features_hack(
            //const image_type& img_, 
            const HogBackend &fabHog,
            out_type& hog, 
            int cell_size,
            int filter_rows_padding,
            int filter_cols_padding
        ) 
        {
            //const_image_view<image_type> img(img_);
            // make sure requires clause is not broken
            DLIB_ASSERT( cell_size > 0 &&
                         filter_rows_padding > 0 &&
                         filter_cols_padding > 0 ,
                "\t void extract_fhog_features()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t cell_size: " << cell_size 
                << "\n\t filter_rows_padding: " << filter_rows_padding 
                << "\n\t filter_cols_padding: " << filter_cols_padding 
                );

            /*
                This function implements the HOG feature extraction method described in 
                the paper:
                    P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
                    Object Detection with Discriminatively Trained Part Based Models
                    IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010

                Moreover, this function is derived from the HOG feature extraction code
                from the features.cc file in the voc-releaseX code (see
                http://people.cs.uchicago.edu/~rbg/latent/) which is has the following
                license (note that the code has been modified to work with grayscale and
                color as well as planar and interlaced input and output formats):

                Copyright (C) 2011, 2012 Ross Girshick, Pedro Felzenszwalb
                Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
                Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan

                Permission is hereby granted, free of charge, to any person obtaining
                a copy of this software and associated documentation files (the
                "Software"), to deal in the Software without restriction, including
                without limitation the rights to use, copy, modify, merge, publish,
                distribute, sublicense, and/or sell copies of the Software, and to
                permit persons to whom the Software is furnished to do so, subject to
                the following conditions:

                The above copyright notice and this permission notice shall be
                included in all copies or substantial portions of the Software.

                THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
                EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
                MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
                NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
                LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
                OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
                WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
            */

            // if (cell_size == 1)
            // {
            //     impl_extract_fhog_features_hack_cell_size_1(fabHog,hog,filter_rows_padding,filter_cols_padding);
            //     return;
            // }

            // unit vectors used to compute gradient orientation
            // matrix<float,2,1> directions[9];
            // directions[0] =  1.0000, 0.0000; 
            // directions[1] =  0.9397, 0.3420;
            // directions[2] =  0.7660, 0.6428;
            // directions[3] =  0.500,  0.8660;
            // directions[4] =  0.1736, 0.9848;
            // directions[5] = -0.1736, 0.9848;
            // directions[6] = -0.5000, 0.8660;
            // directions[7] = -0.7660, 0.6428;
            // directions[8] = -0.9397, 0.3420;



            // // First we allocate memory for caching orientation histograms & their norms.
            // const int cells_nr = (int)((float)img.nr()/(float)cell_size + 0.5);
            // const int cells_nc = (int)((float)img.nc()/(float)cell_size + 0.5);

            // if (cells_nr == 0 || cells_nc == 0)
            // {
            //     hog.clear();
            //     return;
            // }

            // // We give hist extra padding around the edges (1 cell all the way around the
            // // edge) so we can avoid needing to do boundary checks when indexing into it
            // // later on.  So some statements assign to the boundary but those values are
            // // never used.
            // array2d<matrix<float,18,1> > hist(cells_nr+2, cells_nc+2);
            // for (long r = 0; r < hist.nr(); ++r)
            // {
            //     for (long c = 0; c < hist.nc(); ++c)
            //     {
            //         hist[r][c] = 0;
            //     }
            // }

            // array2d<float> norm(cells_nr, cells_nc);
            // assign_all_pixels(norm, 0);

            // // memory for HOG features
            // const int hog_nr = std::max(cells_nr-2, 0);
            // const int hog_nc = std::max(cells_nc-2, 0);
            // if (hog_nr == 0 || hog_nc == 0)
            // {
            //     hog.clear();
            //     return;
            // }
            // const int padding_rows_offset = (filter_rows_padding-1)/2;
            // const int padding_cols_offset = (filter_cols_padding-1)/2;
            // init_hog(hog, hog_nr, hog_nc, filter_rows_padding, filter_cols_padding);

            // const int visible_nr = std::min((long)cells_nr*cell_size,img.nr())-1;
            // const int visible_nc = std::min((long)cells_nc*cell_size,img.nc())-1;

            // // First populate the gradient histograms
            // for (int y = 1; y < visible_nr; y++) 
            // {
            //     const float yp = ((float)y+0.5)/(float)cell_size - 0.5;
            //     const int iyp = (int)std::floor(yp);
            //     const float vy0 = yp - iyp;
            //     const float vy1 = 1.0 - vy0;
            //     int x;
            //     for (x = 1; x < visible_nc - 7; x += 8)
            //     {
            //         simd8f xx(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
            //         // v will be the length of the gradient vectors.
            //         simd8f grad_x, grad_y, v;
            //         get_gradient(y, x, img, grad_x, grad_y, v);

            //         // We will use bilinear interpolation to add into the histogram bins.
            //         // So first we precompute the values needed to determine how much each
            //         // pixel votes into each bin.
            //         simd8f xp = (xx + 0.5) / (float)cell_size + 0.5;
            //         simd8i ixp = simd8i(xp);
            //         simd8f vx0 = xp - ixp;
            //         simd8f vx1 = 1.0f - vx0;

            //         v = sqrt(v);

            //         // Now snap the gradient to one of 18 orientations
            //         simd8f best_dot = 0;
            //         simd8f best_o = 0;
            //         for (int o = 0; o < 9; o++)
            //         {
            //             simd8f dot = grad_x*directions[o](0) + grad_y*directions[o](1);
            //             simd8f_bool cmp = dot>best_dot;
            //             best_dot = select(cmp, dot, best_dot);
            //             dot *= -1;
            //             best_o = select(cmp, o, best_o);

            //             cmp = dot > best_dot;
            //             best_dot = select(cmp, dot, best_dot);
            //             best_o = select(cmp, o + 9, best_o);
            //         }


            //         // Add the gradient magnitude, v, to 4 histograms around pixel using
            //         // bilinear interpolation.
            //         vx1 *= v;
            //         vx0 *= v;
            //         // The amounts for each bin
            //         simd8f v11 = vy1*vx1;
            //         simd8f v01 = vy0*vx1;
            //         simd8f v10 = vy1*vx0;
            //         simd8f v00 = vy0*vx0;

            //         int32 _best_o[8]; simd8i(best_o).store(_best_o);
            //         int32 _ixp[8];    ixp.store(_ixp);
            //         float _v11[8];    v11.store(_v11);
            //         float _v01[8];    v01.store(_v01);
            //         float _v10[8];    v10.store(_v10);
            //         float _v00[8];    v00.store(_v00);

            //         hist[iyp + 1][_ixp[0]](_best_o[0]) += _v11[0];
            //         hist[iyp + 1 + 1][_ixp[0]](_best_o[0]) += _v01[0];
            //         hist[iyp + 1][_ixp[0] + 1](_best_o[0]) += _v10[0];
            //         hist[iyp + 1 + 1][_ixp[0] + 1](_best_o[0]) += _v00[0];

            //         hist[iyp + 1][_ixp[1]](_best_o[1]) += _v11[1];
            //         hist[iyp + 1 + 1][_ixp[1]](_best_o[1]) += _v01[1];
            //         hist[iyp + 1][_ixp[1] + 1](_best_o[1]) += _v10[1];
            //         hist[iyp + 1 + 1][_ixp[1] + 1](_best_o[1]) += _v00[1];

            //         hist[iyp + 1][_ixp[2]](_best_o[2]) += _v11[2];
            //         hist[iyp + 1 + 1][_ixp[2]](_best_o[2]) += _v01[2];
            //         hist[iyp + 1][_ixp[2] + 1](_best_o[2]) += _v10[2];
            //         hist[iyp + 1 + 1][_ixp[2] + 1](_best_o[2]) += _v00[2];

            //         hist[iyp + 1][_ixp[3]](_best_o[3]) += _v11[3];
            //         hist[iyp + 1 + 1][_ixp[3]](_best_o[3]) += _v01[3];
            //         hist[iyp + 1][_ixp[3] + 1](_best_o[3]) += _v10[3];
            //         hist[iyp + 1 + 1][_ixp[3] + 1](_best_o[3]) += _v00[3];

            //         hist[iyp + 1][_ixp[4]](_best_o[4]) += _v11[4];
            //         hist[iyp + 1 + 1][_ixp[4]](_best_o[4]) += _v01[4];
            //         hist[iyp + 1][_ixp[4] + 1](_best_o[4]) += _v10[4];
            //         hist[iyp + 1 + 1][_ixp[4] + 1](_best_o[4]) += _v00[4];

            //         hist[iyp + 1][_ixp[5]](_best_o[5]) += _v11[5];
            //         hist[iyp + 1 + 1][_ixp[5]](_best_o[5]) += _v01[5];
            //         hist[iyp + 1][_ixp[5] + 1](_best_o[5]) += _v10[5];
            //         hist[iyp + 1 + 1][_ixp[5] + 1](_best_o[5]) += _v00[5];

            //         hist[iyp + 1][_ixp[6]](_best_o[6]) += _v11[6];
            //         hist[iyp + 1 + 1][_ixp[6]](_best_o[6]) += _v01[6];
            //         hist[iyp + 1][_ixp[6] + 1](_best_o[6]) += _v10[6];
            //         hist[iyp + 1 + 1][_ixp[6] + 1](_best_o[6]) += _v00[6];

            //         hist[iyp + 1][_ixp[7]](_best_o[7]) += _v11[7];
            //         hist[iyp + 1 + 1][_ixp[7]](_best_o[7]) += _v01[7];
            //         hist[iyp + 1][_ixp[7] + 1](_best_o[7]) += _v10[7];
            //         hist[iyp + 1 + 1][_ixp[7] + 1](_best_o[7]) += _v00[7];
            //     }
            //     // Now process the right columns that don't fit into simd registers.
            //     for (; x < visible_nc; x++) 
            //     {
            //         matrix<float, 2, 1> grad;
            //         float v;
            //         get_gradient(y,x,img,grad,v);

            //         // snap to one of 18 orientations
            //         float best_dot = 0;
            //         int best_o = 0;
            //         for (int o = 0; o < 9; o++) 
            //         {
            //             const float dot = dlib::dot(directions[o], grad);
            //             if (dot > best_dot) 
            //             {
            //                 best_dot = dot;
            //                 best_o = o;
            //             } 
            //             else if (-dot > best_dot) 
            //             {
            //                 best_dot = -dot;
            //                 best_o = o+9;
            //             }
            //         }

            //         v = std::sqrt(v);
            //         // add to 4 histograms around pixel using bilinear interpolation
            //         const float xp = ((double)x + 0.5) / (double)cell_size - 0.5;
            //         const int ixp = (int)std::floor(xp);
            //         const float vx0 = xp - ixp;
            //         const float vx1 = 1.0 - vx0;

            //         hist[iyp+1][ixp+1](best_o) += vy1*vx1*v;
            //         hist[iyp+1+1][ixp+1](best_o) += vy0*vx1*v;
            //         hist[iyp+1][ixp+1+1](best_o) += vy1*vx0*v;
            //         hist[iyp+1+1][ixp+1+1](best_o) += vy0*vx0*v;
            //     }
            // }

            const int cells_nr = fabHog.extractHeight;
            const int cells_nc = fabHog.extractWidth;

            const int hog_nr = std::max(cells_nr-2, 0);
            const int hog_nc = std::max(cells_nc-2, 0);

            const int padding_rows_offset = 0;
            const int padding_cols_offset = 0;

            init_hog(hog, hog_nr, hog_nc, filter_rows_padding, filter_cols_padding);

            array2d<float> norm(cells_nr, cells_nc);
            assign_all_pixels(norm, 0);


            std::cerr << "HISTO_CELL_SIZE=" << HISTO_CELL_SIZE << "\n";
            std::cerr << "HISTO_BIN_WIDTH_SHIFT=" << HISTO_BIN_WIDTH_SHIFT << "\n";
            std::cerr << "HISTO_BIN_WIDTH=" << HISTO_BIN_WIDTH << "\n";
            std::cerr << "HISTO_BIN_WIDTH_MASK=" << HISTO_BIN_WIDTH_MASK << "\n";
            std::cerr << "HISTO_BIN_COUNT_SHIFT=" << HISTO_BIN_COUNT_SHIFT << "\n";
            std::cerr << "HISTO_BIN_COUNT=" << HISTO_BIN_COUNT << "\n";

            array2d<matrix<float,HISTO_BIN_COUNT,1> > hist(cells_nr, cells_nc);

            for (long r = 0; r < hist.nr(); ++r)
            {
                for (long c = 0; c < hist.nc(); ++c)
                {
                    for (int b=0; b<HISTO_BIN_COUNT; ++b)
                    {
                        hist[r][c](b) = fabHog.extract()[r*cells_nc*HISTO_BIN_COUNT + c*HISTO_BIN_COUNT + b];
                    }
                }
            }
            // for (long r = 0; r < hist.nr(); ++r)
            // {
            //     for (long c = 0; c < hist.nc(); ++c)
            //     {
            //         for (int i=0; i<HISTO_BIN_COUNT; ++i)
            //         {

            //         }
            //         hist[r][c] = fabHog.hist()[r*cells_nc + c];
            //     }
            // }

            // std::cerr << "visible_nr=" << visible_nr << "\n";
            // std::cerr << "visible_nc=" << visible_nc << "\n";
            std::cerr << "cells_nr=" << cells_nr << "\n";
            std::cerr << "cells_nc=" << cells_nc << "\n";
            //std::cerr << "cell_size=" << cell_size << "\n";
            std::cerr << "hog_nr=" << hog_nr << "\n";
            std::cerr << "hog_nc=" << hog_nc << "\n";

            // compute energy in each block by summing over orientations
            for (int r = 0; r < cells_nr; ++r)
            {
                for (int c = 0; c < cells_nc; ++c)
                {
                    for (int o = 0; o < 9; o++) 
                    {
                        norm[r][c] += (hist[r+1][c+1](o) + hist[r+1][c+1](o+9)) * (hist[r+1][c+1](o) + hist[r+1][c+1](o+9));
                    }
                }
            }

            const float eps = 0.0001;
            // compute features
            for (int y = 0; y < hog_nr; y++) 
            {
                const int yy = y+padding_rows_offset; 
                for (int x = 0; x < hog_nc; x++) 
                {
                    const simd4f z1(norm[y+1][x+1],
                                    norm[y][x+1], 
                                    norm[y+1][x],  
                                    norm[y][x]);

                    const simd4f z2(norm[y+1][x+2],
                                    norm[y][x+2],
                                    norm[y+1][x+1],
                                    norm[y][x+1]);

                    const simd4f z3(norm[y+2][x+1],
                                    norm[y+1][x+1],
                                    norm[y+2][x],
                                    norm[y+1][x]);

                    const simd4f z4(norm[y+2][x+2],
                                    norm[y+1][x+2],
                                    norm[y+2][x+1],
                                    norm[y+1][x+1]);

                    const simd4f nn = 0.2*sqrt(z1+z2+z3+z4+eps);
                    const simd4f n = 0.1/nn;

                    simd4f t = 0;

                    const int xx = x+padding_cols_offset; 

                    // contrast-sensitive features
                    for (int o = 0; o < 18; o+=3) 
                    {
                        simd4f temp0(hist[y+1+1][x+1+1](o));
                        simd4f temp1(hist[y+1+1][x+1+1](o+1));
                        simd4f temp2(hist[y+1+1][x+1+1](o+2));
                        simd4f h0 = min(temp0,nn)*n;
                        simd4f h1 = min(temp1,nn)*n;
                        simd4f h2 = min(temp2,nn)*n;
                        set_hog(hog,o,xx,yy,   sum(h0));
                        set_hog(hog,o+1,xx,yy, sum(h1));
                        set_hog(hog,o+2,xx,yy, sum(h2));
                        t += h0+h1+h2;
                    }

                    t *= 2*0.2357;

                    // contrast-insensitive features
                    for (int o = 0; o < 9; o+=3) 
                    {
                        simd4f temp0 = hist[y+1+1][x+1+1](o)   + hist[y+1+1][x+1+1](o+9);
                        simd4f temp1 = hist[y+1+1][x+1+1](o+1) + hist[y+1+1][x+1+1](o+9+1);
                        simd4f temp2 = hist[y+1+1][x+1+1](o+2) + hist[y+1+1][x+1+1](o+9+2);
                        simd4f h0 = min(temp0,nn)*n;
                        simd4f h1 = min(temp1,nn)*n;
                        simd4f h2 = min(temp2,nn)*n;
                        set_hog(hog,o+18,xx,yy, sum(h0));
                        set_hog(hog,o+18+1,xx,yy, sum(h1));
                        set_hog(hog,o+18+2,xx,yy, sum(h2));
                    }


                    float temp[4];
                    t.store(temp);

                    // texture features
                    set_hog(hog,27,xx,yy, temp[0]);
                    set_hog(hog,28,xx,yy, temp[1]);
                    set_hog(hog,29,xx,yy, temp[2]);
                    set_hog(hog,30,xx,yy, temp[3]);
                }
            }
        }

    // ------------------------------------------------------------------------------------

    } // end namespace impl_fhog

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    // template <
    //     typename image_type, 
    //     typename T, 
    //     typename mm1, 
    //     typename mm2
    //     >
    // void extract_fhog_features_hack(
    //     const image_type& img, 
    //     dlib::array<array2d<T,mm1>,mm2>& hog, 
    //     int cell_size = 8,
    //     int filter_rows_padding = 1,
    //     int filter_cols_padding = 1
    // ) 
    // {
    //     impl_fhog::impl_extract_fhog_features_hack(img, hog, cell_size, filter_rows_padding, filter_cols_padding);
    //     // If the image is too small then the above function outputs an empty feature map.
    //     // But to make things very uniform in usage we require the output to still have the
    //     // 31 planes (but they are just empty).
    //     if (hog.size() == 0)
    //         hog.resize(31);
    // }

    //typename image_type, 
    template <
        typename T, 
        typename mm
        >
    void extract_fhog_features_hack(
        //const image_type& img, 
        const HogBackend &fabHog,
        array2d<matrix<T,31,1>,mm>& hog, 
        int cell_size = 8,
        int filter_rows_padding = 1,
        int filter_cols_padding = 1
    ) 
    {
        impl_fhog::impl_extract_fhog_features_hack(fabHog, hog, cell_size, filter_rows_padding, filter_cols_padding);
    }

// ----------------------------------------------------------------------------------------

    // template <
    //     typename image_type,
    //     typename T
    //     >
    // void extract_fhog_features_hack(
    //     const image_type& img, 
    //     matrix<T,0,1>& feats,
    //     int cell_size = 8,
    //     int filter_rows_padding = 1,
    //     int filter_cols_padding = 1
    // )
    // {
    //     dlib::array<array2d<T> > hog;
    //     extract_fhog_features_hack(img, hog, cell_size, filter_rows_padding, filter_cols_padding);
    //     feats.set_size(hog.size()*hog[0].size());
    //     for (unsigned long i = 0; i < hog.size(); ++i)
    //     {
    //         const long size = hog[i].size();
    //         set_rowm(feats, range(i*size, (i+1)*size-1)) = reshape_to_column_vector(mat(hog[i]));
    //     }
    // }
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fHOG_Hh_

