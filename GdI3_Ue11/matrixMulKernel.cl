__kernel void matrixMul( __global float* C, __global float* A, __global float* B, int widthA, int heightA, int widthB)
{
	int i = get_global_id(1);// / widthB;
	int j = get_global_id(0);// % widthB;
	int k;

	C[i*widthB + j] = 0;
	for (k = 0; k < widthA; k++) {
			 C[i*widthB + j] += A[i*widthA + k] * B[k*widthB + j];
	}
	return;
}
