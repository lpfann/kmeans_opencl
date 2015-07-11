
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
            sum += (data[i*dim +d] - prototypes[k*dim+d])*(data[i*dim +d] - prototypes[k*dim+d]);
        }

        min = fmin(min,sum);
        nearestproto = (min==sum) ? k : nearestproto;

    }
    out_assignment[i] = nearestproto;
}

__kernel void calc_dist_to_nearest_prototype(
__global const float* data,
__global const float* prototypes,
__global float* out_dist,
const uint dim,
const uint K,
uint n,
const uint current)
{

    const int i = get_global_id(0);
    if (i >= n)
        return;

    float min = FLT_MAX;
    float sum;
    for (uint k = 0; k < current; k++){

        sum = 0.0f;
        for (uint d = 0; d < dim;d++){
            sum += (data[i*dim +d] - prototypes[k*dim+d]) * (data[i*dim +d] - prototypes[k*dim+d]);
        }

        min = fmin(min,sum);
    }


    out_dist[i] = min;
}


__kernel void calc_prototype(
__global const float* data,
__global const uint* assignment,
__global float* out_prototypes,
const uint dim,
const uint K,
const uint N)
{

    const int i = get_global_id(0);
    if (i >= K*dim)
        return;
    const uint k = i/dim;
    const uint d = i%dim;

    float element = 0.0f;
    uint count = 0;
    for (uint x = 0; x < N; x++){

        if(assignment[x]==k){
            element += data[x*dim +d];
            count++;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // m_k-means method to prohibit empty clusters
    element = (element + out_prototypes[i]) / (count+1);
    barrier(CLK_GLOBAL_MEM_FENCE);
    out_prototypes[i] = element;
}
