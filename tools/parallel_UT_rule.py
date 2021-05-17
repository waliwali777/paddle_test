# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

# *=======These unittest doesn't occupy GPU memory, just run as CPU unittest=======* #
# It run 16 job each time, If it failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED, 
# just remove it from this list.
CPU_PARALLEL_JOB = [
    'test_static_save_load_large',
    'version_test',
    'var_type_traits_test',
    'var_type_inference_test',
    'variable_test',
    'unroll_array_ops_test',
    'tuple_test',
    'to_string_test',
    'timer_test',
    'threadpool_test',
    'test_zeros_op',
    'test_while_op',
    'test_weight_quantization_mobilenetv1',
    'test_version',
    'test_var_info',
    'test_var_conv_2d',
    'test_utils',
    'test_unique_name',
    'test_transpose_int8_mkldnn_op',
    'test_transpose_bf16_mkldnn_op',
    'test_trainer_desc',
    'test_trainable',
    'test_teacher_student_sigmoid_loss_op',
    'test_tdm_sampler_op',
    'test_tdm_child_op',
    'test_sysconfig',
    'test_sync_batch_norm_pass',
    'test_switch',
    'test_static_shape_inferrence_for_shape_tensor',
    'test_static_analysis',
    'test_squared_mat_sub_fuse_pass',
    'test_split_and_merge_lod_tensor_op',
    'test_spawn_and_init_parallel_env',
    'test_slice_var',
    'test_skip_layernorm_fuse_pass',
    'test_simplify_with_basic_ops_pass',
    'test_similarity_focus_op',
    'test_shuffle_batch_op',
    'test_shrink_rnn_memory',
    'test_set_bool_attr',
    'test_sequence_topk_avg_pooling',
    'test_sequence_scatter_op',
    'test_sequence_scatter_op',
    'test_sequence_last_step',
    'test_sequence_first_step',
    'test_seqpool_cvm_concat_fuse_pass',
    'test_seqpool_concat_fuse_pass',
    'test_seq_concat_fc_fuse_pass',
    'test_selected_rows',
    'test_scope',
    'test_scale_matmul_fuse_pass',
    'test_scaled_dot_product_attention',
    'test_sampling_id_op',
    'test_runtime_and_compiletime_exception',
    'test_run_fluid_by_module_or_command_line',
    'test_rpn_target_assign_op',
    'test_row_conv',
    'test_rnn_memory_helper_op',
    'test_retinanet_detection_output',
    'test_reshape_transpose_matmul_mkldnn_fuse_pass',
    'test_reshape_bf16_op',
    'test_require_version',
    'test_requantize_mkldnn_op',
    'test_repeated_fc_relu_fuse_pass',
    'test_repeated_fc_relu_fuse_pass',
    'test_registry',
    'test_reducescatter_api',
    'test_reducescatter',
    'test_recurrent_op',
    'test_recommender_system',
    'test_query_op',
    'test_quantize_transpiler',
    'test_quantize_mkldnn_op',
    'test_quantization_mkldnn_pass',
    'test_quant_int8_resnet50_mkldnn',
    'test_quant_int8_mobilenetv2_mkldnn',
    'test_quant_int8_mobilenetv1_mkldnn',
    'test_quant_int8_googlenet_mkldnn',
    'test_quant2_int8_resnet50_range_mkldnn',
    'test_quant2_int8_resnet50_mkldnn',
    'test_quant2_int8_resnet50_channelwise_mkldnn',
    'test_quant2_int8_mobilenetv1_mkldnn',
    'test_quant2_int8_mkldnn_pass',
    'test_quant2_int8_ernie_mkldnn',
    'test_py_reader_sample_generator',
    'test_py_reader_return_list',
    'test_py_reader_lod_level_share',
    'test_py_reader_error_msg',
    'test_pyramid_hash_op',
    'test_pybind_interface',
    'test_ps_dispatcher',
    'test_prune',
    'test_protobuf_descs',
    'test_protobuf',
    'test_progressbar',
    'test_program_to_string',
    'test_program_code',
    'test_program',
    'test_precision_recall_op',
    'test_post_training_quantization_resnet50',
    'test_post_training_quantization_mobilenetv1',
    'test_post_training_quantization_mnist',
    'test_positive_negative_pair_op',
    'test_paddle_inference_api',
    'test_origin_info',
    'test_op_version',
    'test_op_support_gpu',
    'test_operator_desc',
    'test_operator',
    'test_ones_op',
    'test_npair_loss_op',
    'test_nn_functional_embedding_static',
    'test_name_scope',
    'test_naive_executor',
    'test_multiprocess_dataloader_iterable_dataset_split',
    'test_multiprocess_dataloader_exception',
    'test_multihead_matmul_fuse_pass',
    'test_multi_gru_seq_fuse_pass',
    'test_multi_gru_mkldnn_op',
    'test_multi_gru_fuse_pass',
    'test_multiclass_nms_op',
    'test_mul_int8_mkldnn_op',
    'test_mkldnn_scale_matmul_fuse_pass',
    'test_mkldnn_placement_pass',
    'test_mkldnn_op_nhwc',
    'test_mkldnn_op_inplace',
    'test_mkldnn_matmul_transpose_reshape_fuse_pass',
    'test_mkldnn_matmul_op_output_fuse_pass',
    'test_mkldnn_inplace_pass',
    'test_mkldnn_inplace_fuse_pass',
    'test_mkldnn_cpu_bfloat16_pass',
    'test_mkldnn_conv_concat_relu_mkldnn_fuse_pass',
    'test_mkldnn_conv_bias_fuse_pass',
    'test_mkldnn_conv_activation_fuse_pass',
    'test_mine_hard_examples_op',
    'test_memory_usage',
    'test_matrix_nms_op',
    'test_matmul_transpose_reshape_fuse_pass',
    'test_matmul_mkldnn_op',
    'test_matmul_bf16_mkldnn_op',
    'test_math_op_patch',
    'test_match_matrix_tensor_op',
    'test_lookup_table_dequant_op',
    'test_logging_utils',
    'test_logger',
    'test_lod_tensor_array_ops',
    'test_lod_tensor_array',
    'test_lod_rank_table',
    'test_locality_aware_nms_op',
    'test_load_vars_shape_check',
    'test_load_op_xpu',
    'test_load_op',
    'test_limit_gpu_memory',
    'test_layer_norm_mkldnn_op',
    'test_layer_norm_bf16_mkldnn_op',
    'test_layer',
    'test_lambv2_op',
    'test_is_test_pass',
    'test_ir_skip_layernorm_pass',
    'test_ir_graph',
    'test_io_save_load',
    'test_input_spec',
    'test_infer_shape',
    'test_infer_no_need_buffer_slots',
    'test_inference_model_io',
    'test_inference_api',
    'test_imperative_signal_handler',
    'test_imperative_numpy_bridge',
    'test_imperative_group',
    'test_imperative_decorator',
    'test_imperative_data_loader_process',
    'test_imperative_data_loader_exit_func',
    'test_imperative_base',
    'test_image_classification_layer',
    'test_image',
    'test_ifelse_basic',
    'test_hsigmoid_op',
    'test_hooks',
    'test_hash_op',
    'test_group',
    'test_graph_pattern_detector',
    'test_gpu_package_without_gpu_device',
    'test_global_var_getter_setter',
    'test_get_set_flags',
    'test_generator',
    'test_generate_proposal_labels_op',
    'test_generate_mask_labels_op',
    'test_gast_with_compatibility',
    'test_fusion_squared_mat_sub_op',
    'test_fusion_seqpool_cvm_concat_op',
    'test_fusion_seqpool_concat_op',
    'test_fusion_seqexpand_concat_fc_op',
    'test_fusion_seqconv_eltadd_relu_op',
    'test_fusion_repeated_fc_relu_op',
    'test_fusion_lstm_op',
    'test_fusion_gru_op',
    'test_fusion_gru_mkldnn_op',
    'test_fusion_gru_int8_mkldnn_op',
    'test_fusion_gru_bf16_mkldnn_op',
    'test_fused_emb_seq_pool_op',
    'test_fused_embedding_fc_lstm_op',
    'test_function_spec',
    'test_full_op',
    'test_fs_interface',
    'test_fs',
    'test_framework_debug_str',
    'test_fp16_utils',
    'test_fleet_util',
    'test_fleet_unitaccessor',
    'test_fleet_runtime',
    'test_fleet_rolemaker_init',
    'test_bf16_utils',
    'test_fleet_rolemaker_4',
    'test_fleet_rolemaker_3',
    'test_fleet_rolemaker',
    'test_fleet_nocvm_1',
    'test_fleet_base_4',
    'test_fleet',
    'test_fleet',
    'test_flags_use_mkldnn',
    'test_flags_mkldnn_ops_on_off',
    'test_filter_by_instag_op',
    'test_fetch_var',
    'test_fetch_handler',
    'test_feed_fetch_method',
    'test_fc_mkldnn_op',
    'test_fc_lstm_fuse_pass',
    'test_fc_lstm_fuse_pass',
    'test_fc_gru_fuse_pass',
    'test_fc_gru_fuse_pass',
    'test_fc_elementwise_layernorm_fuse_pass',
    'test_fc_bf16_mkldnn_op',
    'test_executor_feed_non_tensor',
    'test_executor_check_feed',
    'test_executor_and_use_program_cache',
    'test_exception',
    'test_error_clip',
    'test_entry_attr2',
    'test_entry_attr',
    'test_embedding_eltwise_layernorm_fuse_pass',
    'test_elementwise_mul_bf16_mkldnn_op',
    'test_elementwise_add_bf16_mkldnn_op',
    'test_eager_deletion_recurrent_op',
    'test_eager_deletion_padding_rnn',
    'test_eager_deletion_mnist',
    'test_eager_deletion_dynamic_rnn_base',
    'test_eager_deletion_conditional_block',
    'test_dynrnn_static_input',
    'test_dynrnn_gradient_check',
    'test_dyn_rnn',
    'test_dygraph_mode_of_unittest',
    'test_dpsgd_op',
    'test_downpoursgd',
    'test_download',
    'test_distributions',
    'test_distributed_reader',
    'test_directory_migration',
    'test_detection_map_op',
    'test_desc_clone',
    'test_dequantize_mkldnn_op',
    'test_depthwise_conv_mkldnn_pass',
    'test_deprecated_memory_optimize_interfaces',
    'test_default_scope_funcs',
    'test_default_dtype',
    'test_debugger',
    'test_dataset_wmt',
    'test_dataset_voc',
    'test_dataset_uci_housing',
    'test_dataset_movielens',
    'test_dataset_imikolov',
    'test_dataset_imdb',
    'test_dataset_conll05',
    'test_dataset_cifar',
    'test_dataloader_dataset',
    'test_data_generator',
    'test_data_feeder',
    'test_data',
    'test_cyclic_cifar_dataset',
    'test_cudnn_placement_pass',
    'test_crypto',
    'test_crf_decoding_op',
    'test_create_parameter',
    'test_create_op_doc_string',
    'test_create_global_var',
    'test_cpu_quantize_squash_pass',
    'test_cpu_quantize_placement_pass',
    'test_cpu_quantize_pass',
    'test_cpu_bfloat16_placement_pass',
    'test_cpu_bfloat16_pass',
    'test_conv_elementwise_add_mkldnn_fuse_pass',
    'test_conv_concat_relu_mkldnn_fuse_pass',
    'test_conv_bias_mkldnn_fuse_pass',
    'test_conv_batch_norm_mkldnn_fuse_pass',
    'test_conv_activation_mkldnn_fuse_pass',
    'test_conv3d_transpose_layer',
    'test_conv3d_mkldnn_op',
    'test_conv3d_layer',
    'test_conv2d_transpose_layer',
    'test_conv2d_mkldnn_op',
    'test_conv2d_layer',
    'test_conv2d_int8_mkldnn_op',
    'test_conv2d_bf16_mkldnn_op',
    'test_context_manager',
    'test_const_value',
    'test_conditional_block',
    'test_concat_int8_mkldnn_op',
    'test_concat_bf16_mkldnn_op',
    'test_compat',
    'test_common_infer_shape_functions',
    'test_chunk_eval_op',
    'test_check_import_scipy',
    'test_c_comm_init_all_op',
    'test_calc_gradient',
    'test_broadcast_to_op',
    'test_broadcast_shape',
    'test_broadcast_error',
    'test_broadcast',
    'test_bpr_loss_op',
    'test_boxps',
    'test_bipartite_match_op',
    'test_benchmark',
    'test_beam_search_op',
    'test_batch_sampler',
    'test_batch_norm_act_fuse_pass',
    'test_basic_rnn_name',
    'test_attention_lstm_op',
    'test_analyzer',
    'test_allreduce',
    'test_allgather',
    'test_aligned_allocator',
    'system_allocator_test',
    'stringprintf_test',
    'stringpiece_test',
    'split_test',
    'selected_rows_test',
    'selected_rows_functor_test',
    'scope_test',
    'scatter_test',
    'save_quant2_model_resnet50',
    'save_quant2_model_gru',
    'save_quant2_model_ernie',
    'save_load_util_test',
    'save_load_op_test',
    'save_load_combine_op_test',
    'rw_lock_test',
    'retry_allocator_test',
    'reader_test',
    'reader_blocking_queue_test',
    'prune_test',
    'program_desc_test',
    'profiler_test',
    'place_test',
    'pass_test',
    'op_version_registry_test',
    'op_tester',
    'op_proto_maker_test',
    'op_kernel_type_test',
    'operator_test',
    'operator_exception_test',
    'op_debug_string_test',
    'op_compatible_info_test',
    'op_call_stack_test',
    'no_need_buffer_vars_inference_test',
    'node_test',
    'nccl_context_test',
    'mmap_allocator_test',
    'math_function_test',
    'mask_util_test',
    'lod_tensor_test',
    'test_check_abi',
    'lodtensor_printer_test',
    'jit_kernel_test',
    'test_dispatch_jit',
    'inlined_vector_test',
    'init_test',
    'infer_io_utils_tester',
    'graph_to_program_pass_test',
    'graph_test',
    'graph_helper_test',
    'gather_test',
    'gather_op_test',
    'fused_broadcast_op_test',
    'float16_test',
    'exception_holder_test',
    'errors_test',
    'enforce_test',
    'eigen_test',
    'dropout_op_test',
    'dlpack_tensor_test',
    'dist_multi_trainer_test',
    'dim_test',
    'device_worker_test',
    'decorator_test',
    'ddim_test',
    'data_type_test',
    'test_check_error',
    'data_layout_transform_test',
    'cudnn_helper_test',
    'cudnn_desc_test',
    'cpu_vec_test',
    'cpu_info_test',
    'cpu_helper_test',
    'cow_ptr_tests',
    'conditional_block_op_test',
    'cipher_utils_test',
    'check_reduce_rank_test',
    'buffered_allocator_test',
    'broadcast_op_test',
    'bfloat16_test',
    'beam_search_decode_op_test',
    'auto_growth_best_fit_allocator_test',
    'assign_op_test',
    'allocator_facade_frac_flags_test',
    'aes_cipher_test',
    'test_dist_sparse_tensor_load_adagrad',
    'test_dist_mnist_fp16_allreduce',
    'test_dist_mnist_gradient_merge',
    'test_dist_allreduce_op',
    'test_hdfs3',
    'test_parallel_dygraph_se_resnext',
    'test_dist_fleet_ps9',
    'test_dist_fleet_infer',
    'test_dist_se_resnext_sync',
    'test_dist_oneps',
    'test_dist_sparse_load_ps1',
    'test_dist_mnist_batch_merge',
    'test_dist_fleet_ctr',
    'test_dist_fleet_ps10',
    'test_parallel_dygraph_transformer',
    'test_dist_mnist_fleetapi',
    'test_dist_sparse_tensor_load_adam',
    'test_dist_fleet_ps4',
    'test_dist_fleet_heter_program',
    'test_parallel_dygraph_sparse_embedding_over_height',
    'test_hdfs2',
    'test_dist_sharding_save',
    'test_dist_fleet_ps_gpu_ctr',
    'test_dist_mnist_backward_deps',
    'test_dist_fleet_heter_base',
    'test_dist_sparse_tensor_load_sgd',
    'test_new_group',
    'test_dist_mnist_with_program',
    'test_dist_mnist_pg',
    'test_dist_sparse_tensor_load_rmsprop',
    'test_auto_checkpoint2',
    'test_dist_sparse_tensor_load_ftrl',
    'test_dist_fleet_ps6',
    'test_dist_mnist_fleet_save',
    'test_auto_checkpoint1',
    'test_dist_fleet_a_sync_optimizer_sync',
    'test_dist_fleet_ps3',
    'test_dist_se_resnext_nccl',
    'test_parallel_dygraph_mnist',
    'test_auto_checkpoint_multiple',
    'test_dist_fleet_a_sync_optimizer_auto_async',
    'test_pipeline',
    'test_dist_fleet_ps8',
    'test_dist_fleet_sparse_embedding_ctr',
    'test_dist_se_resnext_dgc',
    'test_dist_fleet_ps7',
    'test_dist_fleet_decay',
    'test_dist_fleet_a_sync_optimizer_auto_geo',
    'test_dist_fleet_geo',
    'test_parallel_dygraph_dataparallel',
    'test_hdfs1',
    'test_dist_mnist_dgc_nccl',
    'test_dist_fleet_ctr2',
    'test_parallel_dygraph_unused_variables',
    'test_dist_mnist_multi_comm',
    'test_dist_sparse_tensor_load_momentum',
    'test_gen_nccl_id_op',
    'test_parallel_dygraph_sparse_embedding',
    'test_dist_mnist_ring_allreduce',
    'test_fleet_launch_async',
    'test_dist_fleet_a_sync_optimizer_geo',
    'test_parallel_dygraph_control_flow',
    'test_auto_checkpoint',
    'test_fleet_pipeline_meta_optimizer',
    'test_dist_fleet_heter_ctr',
    'test_fleet_graph_execution_meta_optimizer',
    'test_fleet_run_random_port',
    'test_dist_fleet_ps5',
    'test_dist_fleet_a_sync_optimizer_auto',
    'test_dist_lookup_sparse_table_fuse_ops',
    'test_dist_fleet_a_sync_optimizer_async',
    'test_c_comm_init_op',
    'test_fleet_launch_nproc',
    'test_dist_fleet_simnet',
    'test_auto_checkpoint_dist_basic',
    'test_fleet_launch_cloud',
    'test_dist_fleet_ps',
    'test_dist_op',
    'test_dist_sparse_load_ps0',
    'test_auto_checkpoint3',
    'test_dist_fleet_ps2',
    'test_dist_fleet_grad_clip',
    'test_custom_concat',
    'test_analyzer_transformer_fuse',
    'test_analyzer_seq_pool1_fuse_statis',
    'test_fc_lstm_fuse_pass_cc',
    'test_layer_norm_fuse_pass',
    'test_fc_gru_fuse_pass_cc',
    'test_analyzer_save_model',
    'test_fleet_ps',
    'test_analyzer_multi_model_prediction',
    'test_fleet_base_3',
    'test_fleet_base_2',
    'test_ascend_trigger',
    'test_fleet_amp_meta_optimizer',
    'test_fleetrun',
    'test_check_abi',
    'dense_table_test',
    'test_adaptive_pool2d_convert_global_pass',
    'test_fleet_recompute_meta_optimizer',
    'test_fleet_fp16_allreduce_meta_optimizer',
    'test_post_training_quantization_lstm_model',
    'test_fleet_metric',
    'test_fleet_gradient_merge_meta_optimizer',
    'test_fleet_sharding_meta_optimizer',
    'test_listen_and_serv_op',
    'test_analyzer_zerocopytensor_tensor',
    'test_conv_bn_fuse_pass_cc',
    'test_collective_optimizer',
    'test_bf16_utils',
    'test_analyzer_seq_pool1_compare_determine',
    'test_avoid_twice_initialization',
    'test_callback_early_stop',
    'test_fleet_distributed_strategy',
    'test_launch_coverage',
    'test_sgd_op_bf16',
    'test_model_cast_to_bf16',
    'test_hybrid_parallel_topology',
    'barrier_table_test',
    'test_check_error',
    'test_fleet_lamb_meta_optimizer',
    'test_fleet_rolemaker_2',
    'test_distributed_strategy',
    'test_rnn_cudnn_params_packing',
    'test_communicator_async',
    'brpc_utils_test',
    'test_analyzer_capi_pd_tensor',
    'test_recv_save_op',
    'heter_listen_and_server_test',
    'test_analyzer_capi_ner',
    'test_unsqueeze2_eltwise_fuse_pass',
    'test_dgc_optimizer',
    'test_fleet_cc',
    'test_repeated_fc_relu_fuse_pass_cc',
    'heter_server_test',
    'test_custom_conj',
    'test_fleet_private_function',
    'test_fake_init_op',
    'brpc_service_sparse_sgd_test',
    'test_tf32_cudnn',
    'test_communicator_geo',
    'test_dispatch_jit',
    'test_layer_norm_fuse_pass_cc',
    'test_fleet_dgc_meta_optimizer',
    'test_fc_fuse_pass_cc',
    'test_communicator_sync',
    'test_analyzer_capi',
    'test_fleet_lars_meta_optimizer',
    'test_communicator_half_async',
    'test_fleet_localsgd_meta_optimizer',
    'test_fleet_amp_init',
    'test_fleet_checkpoint',
    'test_analyzer_seq_pool1_fuse_compare_zero_copy',
    'test_lookup_table_bf16_op',
    'test_fleet_meta_optimizer_base',
    'table_test',
    'test_fleet_rolemaker_new',
    'test_fleet_graph_executor',
    'test_multi_out_jit',
    'test_fleet_utils',
    'brpc_service_dense_sgd_test',
]

# It run 4 job each time, If it failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED,
# just remove it from this list.
TETRAD_PARALLEL_JOB = [
    'graph_node_test',
    'test_assert',
    'test_nce',
    'buffered_allocator_test',
    'allocator_facade_frac_flags_test',
    'cuda_helper_test',
    'test_auto_growth_gpu_memory_limit',
    'device_context_test',
    'test_reference_count_pass_last_lived_ops',
    'copy_same_tensor_test',
    'mixed_vector_test',
    'op_registry_test',
    'test_prepare_op',
    'data_device_transform_test',
    'test_naive_best_fit_gpu_memory_limit',
    'test_imperative_using_non_zero_gpu',
    'retry_allocator_test',
    'system_allocator_test',
    'test_fc_fuse_pass_cc',
    'test_fc_lstm_fuse_pass_cc',
    'test_fc_gru_fuse_pass_cc',
    'test_conv_bn_fuse_pass_cc',
    'test_adaptive_pool2d_convert_global_pass',
    'test_unsqueeze2_eltwise_fuse_pass',
    'test_layer_norm_fuse_pass_cc',
    'test_fc_act_mkldnn_fuse_pass',
    'test_fleet_cc',
    'tensor_test',
    'test_repeated_fc_relu_fuse_pass_cc',
    'test_mkldnn_caching',
    'test_analyzer_seq_pool1',
    'test_analyzer_ocr',
    'test_analyzer_seq_conv1',
    'test_analyzer_small_dam',
    'test_analyzer_mobilenet_depthwise_conv',
    'test_analyzer_pyramid_dnn',
    'test_analyzer_text_classification',
    'test_analyzer_rnn2',
    'test_analyzer_transformer',
    'test_analyzer_resnet50',
    'test_analyzer_ner',
    'test_analyzer_lac',
    'test_analyzer_transformer_profile',
    'test_analyzer_mobilenet_transpose',
    'test_analyzer_rnn1',
    'test_analyzer_seq_pool1_profile',
    'test_analyzer_paddletensor_tensor',
    'test_analyzer_bert',
    'test_analyzer_googlenet',
    'test_fleet_base',
    'test_imperative_container_layerdict',
    'test_set_value_op',
    'test_view_op_reuse_allocation',
    'test_sequential',
    'test_sequential',
    'test_imperative_layers',
    'test_dgc_momentum_op',
    'test_memcpy_op',
    'test_dgc_op',
    'test_lookahead',
    'test_callback_visualdl',
    'test_new_group_api',
    'test_collective_split_embedding_none_divisible',
    'test_collective_wait',
    'test_collective_split_row_linear',
    'test_collective_split_col_linear',
    'test_collective_split_embedding',
    'test_custom_attrs_jit',
    'float16_gpu_test',
    'test_leaky_relu_grad_grad_functor',
    'test_complex_simplenet',
    'selected_rows_functor_gpu_test',
    'test_imperative_framework',
]

# It run 2 job each time, If it failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED,
# just remove it from this list.
TWO_PARALLEL_JOB = [
    'test_tensor_to_numpy',
    'zero_copy_tensor_test',
    'sequence_pooling_test',
    'sequence_padding_test',
    'vol2col_test',
    'convert_model2dot_ernie',
    'im2col_test',
    'test_logical_op',
    'test_imperative_mnist',
    'test_imperative_deepcf',
    'test_cholesky_op',
    'test_sample_logits_op',
    'test_ir_fc_fuse_pass',
    'test_fleet_base_single',
    'test_multiprocess_dataloader_iterable_dataset_dynamic',
    'test_fill_op',
    'test_slice_op',
    'test_cond',
    'test_lstm',
    'test_ema',
    'test_nan_inf',
    'test_isinstance',
    'test_jit_save_load',
    'test_box_clip_op',
    'test_group_norm_op',
    'test_seed_op',
    'test_activation_nn_grad',
    'test_pool2d_int8_mkldnn_op',
    'test_adagrad_op_v2',
    'test_nn_functional_hot_op',
    'test_op_name_conflict',
    'test_imperative_gan',
    'test_simnet',
    'test_amp_check_finite_and_scale_op',
    'test_random_seed',
    'test_histogram_op',
    'test_sequence_conv',
    'test_eye_op',
    'test_row_conv_op',
    'test_full_like_op',
    'test_optimizer_in_control_flow',
    'test_gru_unit_op',
    'test_distribute_fpn_proposals_op',
    'test_log_loss_op',
    'test_adadelta_op',
    'test_diag_embed',
    'test_unsqueeze2_op',
    'test_fused_fc_elementwise_layernorm_op',
    'test_sum_bf16_mkldnn_op',
    'test_sequence_erase_op',
    'test_sigmoid_cross_entropy_with_logits_op',
    'test_regularizer_api',
    'test_lrn_op',
    'test_parallel_ssa_graph_inference_feed_partial_data',
    'test_lod_reset_op',
    'test_install_check',
    'test_anchor_generator_op',
    'test_imperative_ptb_rnn',
    'test_gather_nd_op',
    'test_network_with_dtype',
    'test_elementwise_sub_op',
    'test_assert_op',
    'test_elementwise_div_op',
    'test_gather_tree_op',
    'test_imperative_named_members',
    'test_seqconv_eltadd_relu_fuse_pass',
    'test_analysis_predictor',
    'test_convert_operators',
    'test_add_reader_dependency',
    'test_is_tensor',
    'test_variable',
    'test_save_model_without_var',
    'test_unfold_op',
    'test_conv_bn_fuse_pass',
    'test_truncated_gaussian_random_op',
    'test_tree_conv_op',
    'test_traced_layer_err_msg',
    'test_unique_with_counts',
    'test_auc_single_pred_op',
    'test_conv_bn_fuse_pass',
    'test_instance_norm_op_v2',
    'test_softmax_bf16_mkldnn_op',
    'test_mean_iou',
    'test_sequence_slice_op',
    'test_polygon_box_transform',
    'test_sequence_pad_op',
    'test_sequence_expand',
    'test_cudnn_grucell',
    'test_pool2d_bf16_mkldnn_op',
    'test_bilinear_api',
    'test_parallel_executor_inference_feed_partial_data',
    'test_initializer_nn',
    'test_modified_huber_loss_op',
    'test_lookup_table_op',
    'test_conv1d_layer',
    'test_kron_op',
    'test_isfinite_v2_op',
    'test_ctc_align',
    'test_imperative_save_load_v2',
    'test_decayed_adagrad_op',
    'test_dropout_op',
    'test_functional_conv3d',
    'test_flatten2_op',
    'test_fsp_op',
    'test_fusion_transpose_flatten_concat_op',
    'test_elementwise_nn_grad',
    'test_hinge_loss_op',
    'test_elementwise_add_mkldnn_op',
    'test_optimizer',
    'test_deformable_conv_op',
    'test_py_reader_push_pop',
    'test_random_crop_op',
    'test_shuffle_channel_op',
    'test_center_loss',
    'test_temporal_shift_op',
    'test_case',
    'test_transformer_api',
    'test_adagrad_op',
    'test_batch_norm_mkldnn_op',
    'test_adam_op_multi_thread',
    'test_adamax_op',
    'test_while_loop_op',
    'test_transpose_flatten_concat_fuse_pass',
    'test_trace_op',
    'test_backward',
    'test_top_k_op',
    'test_batch_fc_op',
    'test_tensor_scalar_type_promotion_static',
    'test_squared_l2_distance_op',
    'test_bicubic_interp_op',
    'test_spp_op',
    'test_space_to_depth_op',
    'test_callbacks',
    'test_sigmoid_focal_loss_op',
    'test_collect_fpn_proposals_op',
    'test_sgd_op',
    'test_sequence_unpad_op',
    'test_conv1d_transpose_layer',
    'test_sequence_slice_op',
    'test_sequence_pool',
    'test_conv_elementwise_add_fuse_pass',
    'test_sequence_pad_op',
    'test_conv_shift_op',
    'test_sequence_expand_as',
    'test_cos_sim_op',
    'test_sequence_enumerate_op',
    'test_cross_entropy2_op',
    'test_sequence_concat',
    'test_cudnn_lstmcell',
    'test_data_norm_op',
    'test_decoupled_py_reader_data_check',
    'test_deformable_conv_v1_op',
    'test_roi_align_op',
    'test_detach',
    'test_rnn_cells',
    'test_elementwise_floordiv_op',
    'test_elementwise_min_op',
    'test_reduce_op',
    'test_embedding_id_stop_gradient',
    'test_empty_op',
    'test_py_reader_combination',
    'test_ptb_lm',
    'test_expand_op',
    'test_prroi_pool_op',
    'test_fake_dequantize_op',
    'test_fetch_feed',
    'test_prelu_op',
    'test_fill_zeros_like_op',
    'test_pool2d_op',
    'test_for_enumerate',
    'test_gather_op',
    'test_partial_concat_op',
    'test_gaussian_random_op',
    'test_generate_proposals_v2_op',
    'test_pad_constant_like',
    'test_grid_sample_function',
    'test_pad2d_op',
    'test_huber_loss_op',
    'test_one_hot_op',
    'test_normal',
    'test_imperative_auto_prune',
    'test_nn_grad',
    'test_nearest_interp_op',
    'test_minus_op',
    'test_imperative_reinforcement',
    'test_maxout_op',
    'test_matmul_op',
    'test_increment',
    'test_masked_select_op',
    'test_lstmp_op',
    'test_loop',
    'test_label_smooth_op',
    'test_logsumexp',
    'test_log_softmax',
    'test_learning_rate_scheduler',
    'test_linspace',
    'test_linear_interp_op',
    'test_layer_norm_op_v2',
    'test_lamb_op',
    'test_lookup_table_v2_op',
    'test_l1_norm_op',
    'test_lstm_op',
    'test_margin_rank_loss_op',
    'test_index_sample_op',
    'test_imperative_static_runner_while',
    'test_imperative_save_load',
    'test_imperative_ptb_rnn_sorted_gradient',
    'test_mul_op',
    'test_imperative_lod_tensor_to_selected_rows',
    'test_imperative_data_parallel',
    'test_norm_nn_grad',
    'test_im2sequence_op',
    'test_if_else_op',
    'test_one_hot_v2_op',
    'test_grid_sampler_op',
    'test_pad_op',
    'test_generate_proposals_op',
    'test_parameter',
    'test_gaussian_random_mkldnn_op',
    'test_partial_sum_op',
    'test_ftrl_op',
    'test_flip',
    'test_pool_max_op',
    'test_prior_box_op',
    'test_fake_quantize_op',
    'test_proximal_gd_op',
    'test_expand_v2_op',
    'test_psroi_pool_op',
    'test_expand_as_v2_op',
    'test_ptb_lm_v2',
    'test_rand_op',
    'test_empty_like_op',
    'test_rank_loss_op',
    'test_elementwise_mod_op',
    'test_reinforcement_learning',
    'test_elementwise_max_op',
    'test_retain_graph',
    'test_edit_distance_op',
    'test_reverse_op',
    'test_device_guard',
    'test_rnn_cells_static',
    'test_deformable_psroi_pooling',
    'test_roi_perspective_transform_op',
    'test_segment_ops',
    'test_cvm_op',
    'test_selu_op',
    'test_cross_op',
    'test_sequence_conv',
    'test_crop_tensor_op',
    'test_sequence_expand',
    'test_sequence_mask',
    'test_conv_nn_grad',
    'test_sequence_pool',
    'test_conv_elementwise_add2_act_fuse_pass',
    'test_sequence_reshape',
    'test_conv2d_fusion_op',
    'test_sequence_softmax_op',
    'test_sequence_unpad_op',
    'test_compare_reduce_op',
    'test_clip_by_norm_op',
    'test_box_coder_op',
    'test_smooth_l1_loss_op',
    'test_bilinear_interp_op',
    'test_spectral_norm_op',
    'test_sum_mkldnn_op',
    'test_batch_norm_op',
    'test_base_layer',
    'test_argsort_op',
    'test_arg_min_max_op',
    'test_transpose_op',
    'test_affine_grid_op',
    'test_unpool_op',
    'test_addmm_op',
    'test_adam_optimizer_fp32_fp64',
    'test_auc_op',
    'test_adam_op',
    'test_adam_infcheck_op',
    'test_bilinear_tensor_product_op',
    'test_transpose_mkldnn_op',
    'test_callback_reduce_lr_on_plateau',
    'test_cast_op',
    'test_scatter_nd_op',
    'test_conv2d_transpose_op_depthwise_conv',
    'test_queue',
    'test_cross_entropy_op',
    'test_detection',
    'test_elementwise_mul_mkldnn_op',
    'test_grid_generator',
    'test_functional_conv2d',
    'test_fit_a_line',
    'test_fill_any_like_op',
    'test_functional_conv2d_transpose',
    'test_functional_conv3d_transpose',
    'test_dot_op',
    'test_device',
    'test_imperative_layer_apply',
    'test_dataloader_early_reset',
    'test_imperative_selected_rows_to_lod_tensor',
    'test_crop_op',
    'test_linear_interp_v2_op',
    'test_lr_scheduler',
    'test_tensor_array_to_tensor',
    'test_mean_op',
    'test_momentum_op',
    'test_iou_similarity_op',
    'test_optimizer_grad',
    'test_dygraph_weight_norm',
    'test_batch_norm_op_v2',
    'test_pool2d_mkldnn_op',
    'test_regularizer',
    'test_sequence_concat',
    'test_sequence_expand_as',
    'test_sequence_reverse',
    'test_shape_op',
    'test_diag',
    'test_strided_slice_op',
    'test_switch_case',
    'test_target_assign_op',
    'test_translated_layer',
    'test_isfinite_op',
    'test_conv_elementwise_add_act_fuse_pass',
    'test_unbind_op',
    'test_size_op',
    'test_unique',
    'test_unstack_op',
    'test_wrappers',
    'test_deprecated_decorator',
    'test_affine_channel_op',
    'test_arange',
    'test_lrn_mkldnn_op',
    'test_imperative_gnn',
    'test_dequantize_abs_max_op',
    'test_elementwise_mul_op',
    'test_tensor_scalar_type_promotion_dynamic',
    'test_fc_op',
    'test_mish_op',
    'test_flatten_op',
    'test_gradient_clip',
    'test_allclose_layer',
    'test_meshgrid_op',
    'test_get_places_op',
    'test_reader_reset',
    'test_squared_l2_norm_op',
    'test_softmax_mkldnn_op',
    'test_numel_op',
    'test_squeeze2_op',
    'test_dygraph_mnist_fp16',
    'test_activation_mkldnn_op',
    'test_imperative_layer_children',
    'test_nearest_interp_v2_op',
    'test_fill_zeros_like2_op',
    'test_sync_batch_norm_op',
    'test_static_save_load',
    'test_coalesce_tensor_op',
    'test_fuse_bn_act_pass',
    'test_simnet_v2',
    'test_shard_index_op',
    'test_cuda_random_seed',
    'test_dequantize_log_op',
    'test_mkldnn_batch_norm_act_fuse_pass',
    'test_imperative_skip_op',
    'test_proximal_adagrad_op',
    'test_word2vec',
    'test_conv2d_transpose_mkldnn_op',
    'test_imperative_optimizer',
    'test_assign_value_op',
    'test_roi_pool_op',
    'test_word2vec',
    'test_manual_seed',
    'test_range',
    'test_box_decoder_and_assign_op',
    'test_imperative_optimizer_v2',
    'test_python_operator_overriding',
    'test_is_empty_op',
    'test_py_reader_pin_memory',
    'test_train_recognize_digits',
    'test_parallel_executor_feed_persistable_var',
    'test_mnist',
    'test_update_loss_scaling_op',
    'test_rnn_cell_api',
    'test_imperative_load_static_param',
    'test_fuse_bn_add_act_pass',
    'test_quantize_transpiler_v2',
    'paddle_infer_api_test',
    'test_analyzer_ernie',
    'lite_resnet50_test',
    'lite_mul_model_test',
]


def main():
    cpu_parallel_job = '^job$'
    tetrad_parallel_job = '^job$'
    two_parallel_job = '^job$'
    non_parallel_job = '^job$'

    test_cases = sys.argv[1]
    test_cases = test_cases.split("\n")

    for unittest in CPU_PARALLEL_JOB:
        if unittest in test_cases:
            cpu_parallel_job = cpu_parallel_job + '|^' + unittest + '$'
            test_cases.remove(unittest)

    for unittest in TETRAD_PARALLEL_JOB:
        if unittest in test_cases:
            tetrad_parallel_job = tetrad_parallel_job + '|^' + unittest + '$'
            test_cases.remove(unittest)

    for unittest in TWO_PARALLEL_JOB:
        if unittest in test_cases:
            two_parallel_job = two_parallel_job + '|^' + unittest + '$'
            test_cases.remove(unittest)

    for unittest in test_cases:
        non_parallel_job = non_parallel_job + '|^' + unittest + '$'

    print("{};{};{};{}".format(cpu_parallel_job, tetrad_parallel_job,
                               two_parallel_job, non_parallel_job))


if __name__ == '__main__':
    main()
