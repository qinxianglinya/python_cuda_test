import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
import os
pagelock = 1

module = SourceModule("""
__forceinline__ __device__ float clip(float in, float low, float high)
{
	return (in < low) ? low : (in > high ? high : in);
}
__global__ void YoloResize(unsigned char* input,
	float* output,
	const int outputWidth,
	const int outputHeight,
	const int inputWidth,
	const int inputHeight,
	const int inputChannels)
{
	const int dx = blockIdx.x * blockDim.x + threadIdx.x;
	const int dy = blockIdx.y * blockDim.y + threadIdx.y;
	if ((dx < outputWidth) && (dy < outputHeight))
	{
		if (inputChannels == 1) { 
		}
		else if (inputChannels == 3) {
			double scale_x = (double)inputWidth / outputWidth;
			double scale_y = (double)inputHeight / outputHeight;
			int xmax = outputWidth;
			float fx = (float)((dx + 0.5) * scale_x - 0.5);
			int sx = floor(fx);
			fx = fx - sx;

			int isx1 = sx;
			if (isx1 < 0) {
				fx = 0.0;
				isx1 = 0;
			}
			if (isx1 >= (inputWidth - 1)) {
				xmax = ::min(xmax, dy);
				fx = 0;
				isx1 = inputWidth - 1;
			}

			float2 cbufx;
			cbufx.x = (1.f - fx);
			cbufx.y = fx;

			float fy = (float)((dy + 0.5) * scale_y - 0.5);
			int sy = floor(fy);
			fy = fy - sy;

			int isy1 = clip(sy - 1 + 1 + 0, 0, inputHeight);
			int isy2 = clip(sy - 1 + 1 + 1, 0, inputHeight);

			float2 cbufy;
			cbufy.x = (1.f - fy);
			cbufy.y = fy;

			int isx2 = isx1 + 1;

			float3 d0;

			float3 s11 = make_float3(input[(isy1 * inputWidth + isx1) * inputChannels + 0], input[(isy1 * inputWidth + isx1) * inputChannels + 1], input[(isy1 * inputWidth + isx1) * inputChannels + 2]);
			float3 s12 = make_float3(input[(isy1 * inputWidth + isx2) * inputChannels + 0], input[(isy1 * inputWidth + isx2) * inputChannels + 1], input[(isy1 * inputWidth + isx2) * inputChannels + 2]);
			float3 s21 = make_float3(input[(isy2 * inputWidth + isx1) * inputChannels + 0], input[(isy2 * inputWidth + isx1) * inputChannels + 1], input[(isy2 * inputWidth + isx1) * inputChannels + 2]);
			float3 s22 = make_float3(input[(isy2 * inputWidth + isx2) * inputChannels + 0], input[(isy2 * inputWidth + isx2) * inputChannels + 1], input[(isy2 * inputWidth + isx2) * inputChannels + 2]);

			float h_rst00, h_rst01;

			if (dy > xmax - 1)
			{
				h_rst00 = s11.x;
				h_rst01 = s21.x;
			}
			else
			{
				h_rst00 = s11.x * cbufx.x + s12.x * cbufx.y;
				h_rst01 = s21.x * cbufx.x + s22.x * cbufx.y;
			}

			d0.x = h_rst00 * cbufy.x + h_rst01 * cbufy.y;


			if (dy > xmax - 1)
			{
				h_rst00 = s11.y;
				h_rst01 = s21.y;
			}
			else
			{
				h_rst00 = s11.y * cbufx.x + s12.y * cbufx.y;
				h_rst01 = s21.y * cbufx.x + s22.y * cbufx.y;
			}

			d0.y = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

			if (dy > xmax - 1)
			{
				h_rst00 = s11.z;
				h_rst01 = s21.z;
			}
			else
			{
				h_rst00 = s11.z * cbufx.x + s12.z * cbufx.y;
				h_rst01 = s21.z * cbufx.x + s22.z * cbufx.y;
			}
			d0.z = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

			output[(dy*outputWidth + dx) * 3 + 0] = (d0.x); 
			output[(dy*outputWidth + dx) * 3 + 1] = (d0.y); 
			output[(dy*outputWidth + dx) * 3 + 2] = (d0.z); 
		}
		else {

		}
	}
}
    """)

# block = (32, 32, 1)   blockDim | threadIdx
# grid = (19,19,3))     gridDim  | blockIdx

YoloResizeKer = module.get_function("YoloResize")


def gpu_resize(input_img: np.ndarray, dst_w, dst_h):
    """
    Resize the batch image to (608,608)
    and Convert NHWC to NCHW
    pass the gpu array to normalize the pixel ( divide by 255)
    Application oriented
    input_img : batch input, format: NHWC , recommend RGB. *same as the NN input format
                input must be 3 channel, kernel set ChannelDim as 3.
    out : batch resized array, format: NCHW , same as intput channel
    """
    # ========= Init Params =========
    stream = cuda.Stream()

    # convert to array
    src_h, src_w, channel = input_img.shape
    print(src_h, src_w, channel)

    # Mem Allocation
    # input memory

    if pagelock:  # = = = = = = Pagelock emory = = = = = =
        inp = {"host": cuda.pagelocked_zeros(shape=(src_h, src_w, channel),
                                             dtype=np.uint8,
                                             mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        inp["host"][:src_h, :src_w, :] = input_img

    inp["device"] = cuda.mem_alloc(inp["host"].nbytes)
    cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # output data
    if pagelock:  # = = = = = = Pagelock emory = = = = = =
        out = {"host": cuda.pagelocked_zeros(shape=(dst_h, dst_w, channel),
                                             dtype=np.float32,
                                             mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N H W C

    out["device"] = cuda.mem_alloc(out["host"].nbytes)
    cuda.memcpy_htod_async(out["device"], out["host"], stream)

    pixelGroupSizeX = float(src_w) / float(dst_w)
    pixelGroupSizeY = float(src_h) / float(dst_h)
    print(pixelGroupSizeX)

    # init resize , store kernel in cache
    gridx = (int)((dst_w + 16 - 1) / 16)
    gridy = (int)((dst_h + 16 - 1) / 16)

    YoloResizeKer(inp["device"], out["device"],
                  np.int32(dst_w), np.int32(dst_h),
                  np.int32(src_w), np.int32(src_h),
                  np.int32(channel),
                  block=(16, 16, 1),
                  grid=(gridx, gridy, 1))

    cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    return out["host"]


if __name__ == "__main__":
    batch = 1
    img_batch_0 = np.tile((cv2.imread("d.jpg")), [1, 1, 1])
    a = cv2.imread('d.jpg')
    dstw, dsth = 640, 672
    pix_0 = gpu_resize(img_batch_0, dstw, dsth)
    # print(pix_0[0])
    cv2.imwrite("trans0.jpg", pix_0)
