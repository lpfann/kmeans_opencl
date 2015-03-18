
__kernel void find_nearest_prototype(
__global const float* data,
__global const float* prototypes,
__global int* out_assignment,
const uint dim,
const uint K,
uint n)
{

    int i = get_global_id(0);
    if (i >= n)
        return;

    float min = FLT_MAX ;
    int nearestproto = 0;
    for( int k = 0; k < K;k++){

        float sum = 0.0f;
        for (int d = 0; d < dim;d++){
            sum += native_powr( data[i*dim +d] - prototypes[k*dim+d], 2);
        }

        //Square Root not needed for distance decision
        //sum = native_sqrt(sum);


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