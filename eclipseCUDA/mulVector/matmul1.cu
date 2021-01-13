__global__ void matMul1(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //NxM hilos (distribuidos en bloques de (n x n) hilos
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned id_y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (id_x < d_col) && (id_y < d_fil) ){
        float acc = 0.0;
        //inicializa los punteros a las filas de A y C
        float *rowa = (float *)((char *)d_A + id_y * pitchA);
        float *rowc = (float *)((char *)d_C + id_y * pitchC);
        for (int k = 0; k < d_inner; k++){
            float *rowb = (float *)((char *)d_B + k * pitchB);
            acc = acc + (rowa[k] * rowb[id_x]);
        }
        rowc[id_x] = acc;
    }
}