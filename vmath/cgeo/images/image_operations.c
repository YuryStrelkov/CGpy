#define TRUE 1
#define FALSE 0
#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b

#typedef int(*interplator)(float x, float y, uint8* image, uint rows, uint cols, uint bpp);

uint8 between(float x, float mim, float max)
{
    if(x < min) return FALSE;
    if(x > max) return FALSE;
    return TRUE;
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

int uv_to_pix_local(float x, float y, int cols, int rows,
					int& col, int& row, int& col1, int& row1, float& tx, float& ty)
{
	if(!between(x, 0.0, 1.0))return FALSE;
    if(!between(y, 0.0, 1.0))return FALSE;
	
	col = (int)(x * (cols - 1));
    row = (int)(y * (rows - 1));
	
	col1 = min(col + 1, cols - 1);
    row1 = min(row + 1, rows - 1);
	
    float dx = 1.0 / (cols - 1.0);
    float dy = 1.0 / (rows - 1.0);

    tx = (x - dx * col) / dx;
    ty = (y - dy * row) / dy;
	
	return TRUE
}

int nearest32(float x, float y, uint8* image, uint rows, uint cols, uint bpp)
{
	int col, row, col1, row1;
	float tx, ty;
	
	if (!uv_to_pix_local(x,y, rows, cols, &col, &row, &col1, &row1, &tx, &ty))return 0;

    row = if ty < 0.5 ?row : row1;
    col = if tx < 0.5 ?col : col1;

    col *= bpp;
    row *= bpp;

    if(bpp = 1)
    {
        return  rgb(image[col + row * cols], 0 , 0);
    }
    if(bpp = 3)
    {
        return rgb(image[col + row * cols],
                   image[col + row * cols + 1],
                   image[col + row * cols + 2]);
    }
    if(bpp=4)
    {
        return rgba(image[col + row * cols],
                    image[col + row * cols + 1],
                    image[col + row * cols + 2],
                    image[col + row * cols + 3]);
    }
    return 0;
}

int bilinear32(float x, float y, uint8* image, uint rows, uint cols, int bpp)
{
	int col, row, col1, row1, color = 0;
	float tx, ty;
	if (!uv_to_pix_local(x,y, rows, cols, &col, &row, &col1, &row1, &tx, &ty))return 0;
	
	float q00;
	float q01;
	float q10;
	float q11;
	
	for(int layer = 0; layer < bpp; layer++)
	{
		q00 = (float)(points[(col  + row  * cols) * bpp + layer]);
		q01 = (float)(points[(col1 + row  * cols) * bpp + layer]);
		q10 = (float)(points[(col  + row1 * cols) * bpp + layer]);
		q11 = (float)(points[(col1 + row1 * cols) * bpp + layer]);
		color |= redf(q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11))<<(layer * 8);
	}
    
	return color;
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

float x_derivative(int row, int col, uint8* image, uint rows, uint cols, uint bpp, uint bpp_sift)
{
	int col_prew = max(col-1, 0);
	int col_next = min(col + 1, cols - 1);
	return (image[(row * cols + col_next) * bpp + bpp_sift] -
			image[(row * cols + col_prew) * bpp + bpp_sift]) * 0.5;
};

float y_derivative(int row, int col, uint8* image, uint rows, uint cols, uint bpp, uint bpp_sift)
{
	int row_prew = max(row - 1, 0);
	int row_next = min(row + 1, rows - 1);
	return (image[(row_next * cols + col) * bpp + bpp_sift] -
			image[(row_prew * cols + col) * bpp + bpp_sift]) * 0.5;
};

float xy_derivative(int row, int col, uint8* image, uint rows, uint cols, uint bpp, uint bpp_sift)
{
    int row1 = min(row + 1, rows - 1);
    int row0 = max(0, row - 1);
    
    int col1 = min(col + 1, colons - 1);
    int col0 = max(0, col - 1);
	
	return (image[(row1 * cols + col1) * bpp + bpp_sift] - 
	        image[(row1 * cols + col0) * bpp + bpp_sift]) * 0.25 -
           (image[(row0 * cols + col1) * bpp + bpp_sift] -
   		    image[(row0 * cols + col0) * bpp + bpp_sift]) * 0.25;
}

int bicubic32(float x, float y, uint8* image, uint rows, uint cols, int bpp)
{
	int col, row, col1, row1;
	
	float tx, ty;
	
	if (!uv_to_pix_local(x,y, rows, cols, &col, &row, &col1, &row1, &tx, &ty))return 0;

    float *b = new float[16];
    
	float *c = new float[16];
    
	int row, col, index, color = 0;
	
	for(int layer = 0; i < bpp; layer++)
	{
		b[0 ] = points[(row  * cols + col ) * bpp + layer];
		b[4 ] = points[(row  * cols + col1) * bpp + layer];
		b[8 ] = points[(row1 * cols + col ) * bpp + layer];
		b[12] = points[(row1 * cols + col1) * bpp + layer];
		b[1 ] = x_derivative (row , col , image, rows, cols, bpp, layer);
		b[5 ] = x_derivative (row , col1, image, rows, cols, bpp, layer);
		b[9 ] = x_derivative (row1, col , image, rows, cols, bpp, layer);
		b[13] = x_derivative (row1, col1, image, rows, cols, bpp, layer);
		b[2 ] = y_derivative (row , col , image, rows, cols, bpp, layer);
		b[6 ] = y_derivative (row , col1, image, rows, cols, bpp, layer);
		b[10] = y_derivative (row1, col , image, rows, cols, bpp, layer);
		b[14] = y_derivative (row1, col1, image, rows, cols, bpp, layer);
		b[3 ] = y_derivative (row , col , image, rows, cols, bpp, layer);
		b[7 ] = xy_derivative(row , col1, image, rows, cols, bpp, layer);
		b[11] = xy_derivative(row1, col , image, rows, cols, bpp, layer);
		b[15] = xy_derivative(row1, col1, image, rows, cols, bpp, layer);
		for(index = 0; index < 16; index++)
		{
			 c[index] = 0.0;
		} 
		for(index = 0; index < 256; index++)
		{
			row = index / 16;
			col = index % 16;
			c[row] += _bicubic_poly_coefficients[row * 16 + col] * b[col];
		}
		color |= redf(cubic_poly(tx, ty, c))<<(layer * 8);
	}

    delete c;
    delete b;
	
    return color;
}

void rescale(uint8* src, int src_rows, int src_cols, int src_bpp,
			 uint8* dst, int dst_rows, int dst_cols, int dst_bpp, int method)
{
	int row, col, color;
	float x_col, y_row;
	interplator interp;
	if(method == 0)
	{
		interp = nearest32;
	}else if(method == 1)
	{
		interp = bilinear32;
	}else if(method == 2)
	{
		interp = bicubic32;
	}else
	{
		interp = nearest32;
	}
	int depth, index;
	for(index = 0; i < dst_rows * dst_cols; i++)
	{
		col = index % dst_cols;
		row = index / dst_cols;
		x_col = col * 1.0 / dst_cols;
		y_row = row * 1.0 / dst_rows;
		color = interp(x_col, y_row, src, src_rows, src_cols, src_bpp);
		for(depth = 0; depth < dst_bpp; depth++)
		{
			src[index * dst_bpp + depth] = (uint8)(color|(255 << (8 * depth)))>>(8 * depth);
		}
	}
}