# python_cuda_test
在部署tensrort时发现cv2.resize和cv::cuda::resize输出结果不一致，于是使用pycuda实现resize功能，并使用c cuda完成了相同的resize,使得python上的resize和tensorrt部署时的resize一致。
如有同样需求，可以参考实现。
