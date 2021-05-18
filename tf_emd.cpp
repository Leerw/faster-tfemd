#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>

using namespace tensorflow;

REGISTER_OP("EmdMatch")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("eps: float32")
    .Input("iters: int32")
    .Output("assignment: int32");

REGISTER_OP("EmdCost")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("assignment: int32")
    .Output("dist: float32");

REGISTER_OP("EmdCostGrad")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("grad_cost: float32")
    .Input("assignment: int32")
    .Output("grad_xyz1: float32")
    .Output("grad_xyz2: float32");

int emdMatchLauncher(int b, int n, int m, \
                    const float* xyz1, \
                    const float* xyz2, \
                    int* assignment, \
                    float* price, \
                    int* assignment_inv, \
                    int* bid, \
                    float* bid_increments, \
                    float* max_increments, \
                    int* unass_idx, \
                    int* unass_cnt, \
                    int* unass_cnt_sum, \
                    int* cnt_tmp, \
                    int* max_idx, \
                    const float* eps,
                    const int* iters
);

int emdCostLauncher(int b, int n, const float* xyz1, const float* xyz2, float* dist, const int* assignment);

void emdcostGradLauncher(int b, int n, int m, const float* xyz1, const float* xyz2, const float* grad_cost, const int* assignment, float* grad_xyz1, float* grad_xyz2);


class EmdMatchOp: public OpKernel {
    public:
        explicit EmdMatchOp(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext * context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatch expects (batch_size, num_point, 3) xyz1 shape"));
            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float* xyz1 = &(xyz1_flat(0));

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatch expects (batch_size, num_point, 3) xyz2 shape"));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float* xyz2 = &(xyz2_flat(0));

            const Tensor& eps_tensor = context->input(2);
            OP_REQUIRES(context, eps_tensor.dims() == 1, errors::InvalidArgument("EmdMatch expects constant eps"));
            auto eps_flat = eps_tensor.flat<float>();
            const float* eps = &(eps_flat(0));

            const Tensor& iters_tensor = context->input(3);
            OP_REQUIRES(context, iters_tensor.dims() == 1, errors::InvalidArgument("EmdMatch expects constant iters"));
            auto iters_flat = iters_tensor.flat<int>();
            const int* iters = &(iters_flat(0));
            
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);
            int m = xyz2_tensor.shape().dim_size(1);

            // declare temp tensors
            Tensor assignment_inv_tensor;
            Tensor price_tensor;
            Tensor bid_tensor;
            Tensor bid_increments_tensor;
            Tensor max_increments_tensor;
            Tensor unass_idx_tensor;
            Tensor max_idx_tensor;
            Tensor unass_cnt_tensor;
            Tensor unass_cnt_sum_tensor;
            Tensor cnt_tmp_tensor;

            // allocate temp tensor memory
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{b, m}, &assignment_inv_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{b, m}, &price_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{b, n}, &bid_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{b, n}, &bid_increments_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{b, m}, &max_increments_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{b*n}, &unass_idx_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{b*m}, &max_idx_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{512}, &unass_cnt_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{512}, &unass_cnt_sum_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{512}, &cnt_tmp_tensor));
            
            // initialize assignment_inv
            auto assignment_inv_flat = assignment_inv_tensor.flat<int32>();
            auto price_flat = price_tensor.flat<float>();
            auto bid_flat = bid_tensor.flat<int32>();
            auto bid_increments_flat = bid_increments_tensor.flat<float>();
            auto max_increments_flat = max_increments_tensor.flat<float>();
            auto unass_idx_flat = unass_idx_tensor.flat<int32>();
            auto max_idx_flat = max_idx_tensor.flat<int32>();
            auto unass_cnt_flat = unass_cnt_tensor.flat<int32>();
            auto unass_cnt_sum_flat = unass_cnt_sum_tensor.flat<int32>();
            auto cnt_tmp_flat = cnt_tmp_tensor.flat<int32>();

            int* assignment_inv = &(assignment_inv_flat(0));
            float* price = &(price_flat(0));
            int* bid = &(bid_flat(0));
            float* bid_increments = &(bid_increments_flat(0));
            float* max_increments = &(max_increments_flat(0));
            int* unass_idx = &(unass_idx_flat(0));
            int* max_idx = &(max_idx_flat(0));
            int* unass_cnt = &(unass_cnt_flat(0));
            int* unass_cnt_sum = &(unass_cnt_sum_flat(0));
            int* cnt_tmp = &(cnt_tmp_flat(0));

            // create output tensor
            Tensor* assignment_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n}, &assignment_tensor));
            auto assignment_flat = assignment_tensor->flat<int32>();

            int* assignment = &(assignment_flat(0));

            emdMatchLauncher(b, n, m, xyz1, xyz2, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters);
        }
};

REGISTER_KERNEL_BUILDER(Name("EmdMatch").Device(DEVICE_GPU), EmdMatchOp);


class EmdCostOp: public OpKernel {
    public:
        explicit EmdCostOp(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext * context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatch expects (batch_size, num_point, 3) xyz1 shape"));
            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float* xyz1 = &(xyz1_flat(0));

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatch expects (batch_size, num_point, 3) xyz2 shape"));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float* xyz2 = &(xyz2_flat(0));
            
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& assignment_tensor = context->input(2);
            OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatch expects (batch_size, num_point) assignment shape"));
            auto assignment_flat = assignment_tensor.flat<int32>();
            const int* assignment = &(assignment_flat(0));

            // create output tensor
            Tensor* dist_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n}, &dist_tensor));
            auto dist_flat = dist_tensor->flat<float>();
            float* dist = &(dist_flat(0));

            emdCostLauncher(b, n, xyz1, xyz2, dist, assignment);
        }
};
REGISTER_KERNEL_BUILDER(Name("EmdCost").Device(DEVICE_GPU), EmdCostOp);


class EmdCostGradOp : public OpKernel {
    public:
        explicit EmdCostGradOp (OpKernelConstruction* context) : OpKernel (context) {}
        void Compute(OpKernelContext * context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatchGrad expects (batch_size, num_point, 3) xyz1 shape"));
            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float* xyz1 = &(xyz1_flat(0));

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("EmdMatchGrad expects (batch_size, num_point, 3) xyz2 shape"));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float* xyz2 = &(xyz2_flat(0));

            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);
            int m = xyz2_tensor.shape().dim_size(1);

            const Tensor& grad_cost_tensor = context->input(2);
            OP_REQUIRES(context, grad_cost_tensor.shape() == TensorShape({b, n}), errors::InvalidArgument("EmdMatchGrad expects (batch_size, num_point) grad_cost shape"));
            auto grad_cost_flat = grad_cost_tensor.flat<float>();
            const float* grad_cost = &(grad_cost_flat(0));
            
            const Tensor& assignment_tensor = context->input(3);
            OP_REQUIRES(context, assignment_tensor.shape()==TensorShape({b, n}), errors::InvalidArgument("EmdMatchGrad expects (batchsize, n, m) assignment shape"));
            auto assignment_flat = assignment_tensor.flat<int>();
            const int* assignment = &(assignment_flat(0));

            Tensor* grad_xyz1_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 3}, &grad_xyz1_tensor));
            auto grad_xyz1_flat = grad_xyz1_tensor->flat<float>();
            float* grad_xyz1 = &(grad_xyz1_flat(0));

            Tensor* grad_xyz2_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b, m, 3}, &grad_xyz2_tensor));
            auto grad_xyz2_flat = grad_xyz2_tensor->flat<float>();
            float* grad_xyz2 = &(grad_xyz2_flat(0));

            emdcostGradLauncher(b, n, m, xyz1, xyz2, grad_cost, assignment, grad_xyz1, grad_xyz2);
        }
};
REGISTER_KERNEL_BUILDER(Name("EmdCostGrad").Device(DEVICE_GPU), EmdCostGradOp);
