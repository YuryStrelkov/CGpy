bool between(float x, float mim, float max)
{
    if(x < min) return false;
    if(x > max) return false;
    return true;
};

int red(uint8 r)
{
    return (int)r;
};

int green(uint8 r)
{
    return red(r)<<8;
};

int blue(uint8 r)
{
    return red(r)<<16;
};

int alpha(uint8 r)
{
    return red(r)<<24;
};

int rgb(uint r, uint g, uint b)
{
    return red(r)|green(g)|blue(b);
};

int rgba(uint r, uint g, uint b, uint a)
{
    return red(r)|green(g)|blue(b)|alpha(a);
};

int redf(float r)
{
    return (int)r;
};

int greenf(float r)
{
    return redf(r)<<8;
};

int bluef(float r)
{
    return redf(r)<<16;
};

int alphaf(float r)
{
    return redf(r)<<24;
};

int rgbf(float r, float g, float b)
{
    return redf(r)|greenf(g)|bluef(b);
};

int rgbaf(float r, float g, float b, float a)
{
    return redf(r)|greenf(g)|bluef(b)|alphaf(a);
};

int nearest32(float x, float y, uint8* image, uint rows, uint cols, uint bpp)
{
    if(!between(x, 0.0,1.0))return 0;
    if(!between(y, 0.0,1.0))return 0;

    int col_ = (int)(x * (cols - 1));
    int row_ = (int)(y * (rows - 1));

    float dx_ = 1.0 / (cols - 1.0);
    float dy_ = 1.0 / (rows - 1.0);

    float tx = (x - dx_ * col_) / dx_;
    float ty = (y - dy_ * row_) / dy_;

    row_ = if ty < 0.5 ?row_ : min(row_ + 1, rows - 1);
    col_ = if tx < 0.5 ?col_ : min(col_ + 1, cols - 1);

    col_ *= bpp;
    row_ *= bpp;

    if(bpp = 1)
    {
        return  rgb(image[col_ + row_ * cols], 0 , 0);
    }
    if(bpp = 3)
    {
        return rgb(image[col_ + row_ * cols],
                   image[col_ + row_ * cols + 1],
                   image[col_ + row_ * cols + 2]);
    }
    if(bpp=4)
    {
        return rgba(image[col_ + row_ * cols],
                    image[col_ + row_ * cols + 1],
                    image[col_ + row_ * cols + 2],
                    image[col_ + row_ * cols + 3]);
    }
    return 0;
}

int bilinear32(float x, float y, uint8* image, uint rows, uint cols, uint bpp)
{
    if(!between(x, 0.0,1.0))return 0;
    if(!between(y, 0.0,1.0))return 0;

    int col_ = (int)(x * (cols - 1));
    int row_ = (int)(y * (rows - 1));

    float dx_ = 1.0 / (cols - 1.0);
    float dy_ = 1.0 / (rows - 1.0);

    float tx = (x - dx_ * col_) / dx_;
    float ty = (y - dy_ * row_) / dy_;

    row_ = if ty < 0.5 ?row_ : min(row_ + 1, rows - 1);
    col_ = if tx < 0.5 ?col_ : min(col_ + 1, cols - 1);

    col_ *= bpp;
    row_ *= bpp;

    float q00 = (float)(points[col_  + row_  * cols]);
    float q01 = (float)(points[col_1 + row_  * cols]);
    float q10 = (float)(points[col_  + row_1 * cols]);
    float q11 = (float)(points[col_1 + row_1 * cols]);

    float r =  q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11);

    if(bpp == 1) return rgbf(r, 0.0, 0.0);

    q00 = (float)(points[col_  + row_  * cols + 1]);
    q01 = (float)(points[col_1 + row_  * cols + 1]);
    q10 = (float)(points[col_  + row_1 * cols + 1]);
    q11 = (float)(points[col_1 + row_1 * cols + 1]);

    float g = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11);

    q00 = (float)(points[col_  + row_  * cols + 2]);
    q01 = (float)(points[col_1 + row_  * cols + 2]);
    q10 = (float)(points[col_  + row_1 * cols + 2]);
    q11 = (float)(points[col_1 + row_1 * cols + 2]);

    float b = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11);

    if(bpp == 3)
    {
            return rgbf(r, g, b);
    }

    q00 = float(points[col_  + row_  * cols + 3]);
    q01 = float(points[col_1 + row_  * cols + 3]);
    q10 = float(points[col_  + row_1 * cols + 3]);
    q11 = float(points[col_1 + row_1 * cols + 3]);

    float a = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11);

    return  return rgbaf(r, g, b, a);
}

float cubic_poly(float x, float y, float* coefficients)
{
    float x2 = x * x;
    float x3 = x2 * x;
    float y2 = y * y;
    float y3 = y2 * y;
    return (coefficients[0]  + coefficients[1] *  y + coefficients[2]  * y2 + coefficients[3]  * y3) +
           (coefficients[4]  + coefficients[5] *  y + coefficients[6]  * y2 + coefficients[7]  * y3) * x +
           (coefficients[8]  + coefficients[9] *  y + coefficients[10] * y2 + coefficients[11] * y3) * x2 +
           (coefficients[12] + coefficients[13] * y + coefficients[14] * y2 + coefficients[15] * y3) * x3;
};

int cubic32(float x, float y, uint8* image, uint rows, uint cols, uint bpp)
{
    if(!between(x, 0.0,1.0))return 0;
    if(!between(y, 0.0,1.0))return 0;

    int col_ = (int)(x * (cols - 1));
    int row_ = (int)(y * (rows - 1));

    int col_1 = min(col_ + 1, cols - 1);
    int row_1 = min(row_ + 1, rows - 1);

    float dx_ = 1.0 / (cols - 1.0);
    float dy_ = 1.0 / (rows - 1.0);

    float tx = (x - dx_ * col_) / dx_;
    float ty = (y - dy_ * row_) / dy_;

    row_ = if ty < 0.5 ?row_ : min(row_ + 1, rows - 1);
    col_ = if tx < 0.5 ?col_ : min(col_ + 1, cols - 1);

    col_ *= bpp;
    row_ *= bpp;
    col_1 *= bpp;
    row_1 *= bpp;

    float *b = new float[16];
    float *c = new float[16];
    float dx, dy, dxy;
    int row, col, index;

    /*
        return (points[row, col_1] - points[row, col_0]) * 0.5, \
           (points[row_1, col] - points[row_0, col]) * 0.5, \
           (points[row_1, col_1] - points[row_1, col_0]) * 0.25 - \
           (points[row_0, col_1] - points[row_0, col_0]) * 0.25
    */

    b[0     ]  = points[row_ * cols + col_];
    compute_derivatives_2_at_pt(&dx, &dy, &dxy, points, row_, col_, rows, cols, bpp);
    b[0 + 4 ] = (points[row_ * cols + (col_ + bpp) % (cols * bpp)] - points[row_ * cols, (col_ - bpp) % (cols * bpp)]) * 0.5;
    b[0 + 8 ] = (points[row_1, col] - points[row_0, col]) * 0.5;
    b[0 + 12] = dxy;

    b[1     ]  = points[row_ * cols + col_1];
    compute_derivatives_2_at_pt(&dx, &dy, &dxy, points, row_, col_1, rows, cols, bpp);
    b[1 + 4 ] = dx;
    b[1 + 8 ] = dy;
    b[1 + 12] = dxy;

    b[2     ]  = points[row_1 * cols + col_1];
    compute_derivatives_2_at_pt(&dx, &dy, &dxy, points, row_1, col_, rows, cols, bpp);
    b[2 + 4 ] = dx;
    b[2 + 8 ] = dy;
    b[2 + 12] = dxy;

    b[3     ]  = points[row_1 * cols + col_1];
    compute_derivatives_2_at_pt(&dx, &dy, &dxy, points, row_1, col_1, rows, cols, bpp);
    b[3 + 4 ] = dx;
    b[3 + 8 ] = dy;
    b[3 + 12] = dxy;
    for(index = 0; index < 16; index++) c[index] = 0;
    for(index = 0; index < 256; index++)
    {
        row = index / 16;
        col = index % 16;
        c[row] += _bicubic_poly_coefficients[row * 16 + col] * b[col];
    }

    float r = cubic_poly(tx, ty, c);
    if (bpp = 1)
    {
        delete c;
        delete b;
        return rgbf(r);
    }
}