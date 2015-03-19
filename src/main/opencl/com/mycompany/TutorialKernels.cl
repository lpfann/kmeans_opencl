
__kernel void find_nearest_prototype(
__global const float* data,
__global const float* prototypes,
__global uint* out_assignment,
const uint dim,
const uint K,
uint n)
{

    const int i = get_global_id(0);
    if (i >= n)
        return;

    float min = FLT_MAX;
    
    uint nearestproto = 0;
    float sum;
    for (uint k = 0; k < K; k++){

        sum = 0.0f;
        for (uint d = 0; d < dim;d++){
            sum += native_powr( data[i*dim +d] - prototypes[k*dim+d], 2);
        }

         if (sum < min){
             min = sum;
             nearestproto = k;
         }

    }

    out_assignment[i] = nearestproto;
}

__kernel void calculate_prototype(__global const float* data,__global int* points,__global float* out, const int dim, const int K,int size)
{

    int i = get_global_id(0);
    if (i >= K)
        return;


    for (int s = 0; s < size; s++)  {
        for(int d = 0; d < dim; d++){
            out[d]=data[points[s]+d];
        }
    }
    if(size>0)
        for(int d = 0; d < dim; d++){
            out[d]= out[d]/size;
        }
}