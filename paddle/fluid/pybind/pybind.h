// Generated by the paddle/operator/CMakeLists.txt.  DO NOT EDIT!

USE_CUDA_ONLY_OP(ncclAllReduce);
USE_NO_KERNEL_OP(cond);
USE_OP(cross_entropy);
USE_OP(softmax_with_cross_entropy);
USE_OP(softmax);
USE_OP(detection_output);
USE_OP(sequence_softmax);
USE_OP(sum);
USE_OP(sgd);
USE_NO_KERNEL_OP(print);
USE_OP(adagrad);
USE_OP(maxout);
USE_OP(unpool);
USE_OP(max_pool2d_with_index);
USE_NO_KERNEL_OP(lod_rank_table);
USE_NO_KERNEL_OP(lod_tensor_to_array);
USE_NO_KERNEL_OP(array_to_lod_tensor);
USE_NO_KERNEL_OP(max_sequence_len);
USE_OP(sequence_conv);
USE_OP(sequence_pool);
USE_OP(lstm);
USE_OP(lstmp);
USE_OP(gru);
USE_NO_KERNEL_OP(recurrent);
USE_OP(warpctc);
USE_OP(cos_sim);
USE_NO_KERNEL_OP(parallel_do);
USE_OP(conv2d);
USE_OP(edit_distance);
USE_OP(pool2d);
USE_OP(conv2d_transpose);
USE_OP_DEVICE_KERNEL(conv2d, CUDNN);
USE_OP_DEVICE_KERNEL(pool2d, CUDNN);
USE_OP_DEVICE_KERNEL(conv2d_transpose, CUDNN);
USE_NO_KERNEL_OP(save);
USE_NO_KERNEL_OP(load);
USE_NO_KERNEL_OP(save_combine);
USE_NO_KERNEL_OP(load_combine);
USE_NO_KERNEL_OP(shrink_rnn_memory);
USE_OP(multiplex);
USE_OP(split);
USE_NO_KERNEL_OP(feed);
USE_OP(proximal_gd);
USE_OP(lstm_unit);
USE_NO_KERNEL_OP(merge_lod_tensor);
USE_OP(matmul);
USE_CPU_ONLY_OP(precision_recall);
USE_OP(ctc_align);
USE_OP(crop);
USE_OP(iou_similarity);
USE_OP(scatter);
USE_OP(clip_by_norm);
USE_OP(fill_constant_batch_size_like);
USE_OP(rmsprop);
USE_NO_KERNEL_OP(lod_array_length);
USE_NO_KERNEL_OP(increment);
USE_OP(squared_l2_distance);
USE_NO_KERNEL_OP(get_places);
USE_OP(smooth_l1_loss);
USE_CPU_ONLY_OP(crf_decoding);
USE_OP(bilinear_tensor_product);
USE_OP(scale);
USE_OP(assign_value);
USE_CPU_ONLY_OP(mine_hard_examples);
USE_OP(elementwise_div);
USE_OP(sigmoid_cross_entropy_with_logits);
USE_OP(log_loss);
USE_OP(momentum);
USE_OP(box_coder);
USE_OP(sequence_reshape);
USE_OP(reduce_sum);
USE_OP(split_selected_rows);
USE_OP(decayed_adagrad);
USE_OP(elementwise_sub);
USE_OP(layer_norm);
USE_OP(roi_pool);
USE_NO_KERNEL_OP(while);
USE_NO_KERNEL_OP(is_empty);
USE_CPU_ONLY_OP(nce);
USE_OP(expand);
USE_OP(linear_chain_crf);
USE_OP(sigmoid);
USE_NO_KERNEL_OP(read);
USE_OP(concat);
USE_OP(one_hot);
USE_OP(top_k);
USE_CPU_ONLY_OP(positive_negative_pair);
USE_OP(im2sequence);
USE_CPU_ONLY_OP(chunk_eval);
USE_OP(sequence_expand);
USE_OP(modified_huber_loss);
USE_OP(minus);
USE_OP(huber_loss);
USE_OP(gaussian_random);
USE_OP(elementwise_max);
USE_OP(adamax);
USE_OP(batch_norm);
USE_NO_KERNEL_OP(beam_search);
USE_OP(hinge_loss);
USE_OP(dropout);
USE_OP(row_conv);
USE_OP(conv_shift);
USE_NO_KERNEL_OP(fill);
USE_CPU_ONLY_OP(auc);
USE_OP(ftrl);
USE_NO_KERNEL_OP(fill_constant);
USE_CPU_ONLY_OP(bipartite_match);
USE_OP(spp);
USE_OP(sequence_slice);
USE_OP(sign);
USE_OP(prelu);
USE_OP(mul);
USE_OP(proximal_adagrad);
USE_OP(reshape);
USE_OP(cumsum);
USE_OP(cast);
USE_OP(elementwise_pow);
USE_OP(lookup_table);
USE_OP(label_smooth);
USE_OP(squared_l2_norm);
USE_CPU_ONLY_OP(multiclass_nms);
USE_NO_KERNEL_OP(conditional_block);
USE_OP(adadelta);
USE_OP(gather);
USE_OP(pad);
USE_NO_KERNEL_OP(fetch);
USE_OP(sequence_erase);
USE_OP(uniform_random);
USE_OP(gru_unit);
USE_OP(accuracy);
USE_OP(elementwise_min);
USE_OP(elementwise_add);
USE_OP(fill_zeros_like);
USE_OP(mean);
USE_OP(clip);
USE_OP(rank_loss);
USE_OP(sequence_concat);
USE_NO_KERNEL_OP(assign);
USE_OP(elementwise_mul);
USE_OP(target_assign);
USE_OP(lrn);
USE_OP(margin_rank_loss);
USE_NO_KERNEL_OP(reorder_lod_tensor_by_rank);
USE_NO_KERNEL_OP(beam_search_decode);
USE_NO_KERNEL_OP(rnn_memory_helper);
USE_OP(l1_norm);
USE_NO_KERNEL_OP(split_lod_tensor);
USE_OP(lod_reset);
USE_OP(norm);
USE_OP(adam);
USE_OP(transpose);
USE_CPU_ONLY_OP(prior_box);
USE_OP(less_than);
USE_OP(logical_and);
USE_NO_KERNEL_OP(read_from_array);
USE_NO_KERNEL_OP(create_random_data_generator);
