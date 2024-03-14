def early_stopping(best_loss,loss,best_train,best_test,train_result,test_result, stopping_step, expected_order='acc', flag_step=10):
    # early stopping strategy:
    assert expected_order in ['acc', 'dcc']

    if (expected_order == 'acc' and best_train[2]<train_result[2]) or (expected_order == 'dcc' and loss < best_loss):
        stopping_step = 0
        best_train=train_result
        best_test=test_result
        best_loss=loss
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {}".format(flag_step))
        should_stop = True
    else:
        should_stop = False
    return best_train,best_test,best_loss,stopping_step, should_stop
