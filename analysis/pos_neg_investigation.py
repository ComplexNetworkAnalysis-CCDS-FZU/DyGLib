from utils.DataLoader import get_link_prediction_data



# get data for training, validation and testing
(
    node_raw_features,
    edge_raw_features,
    full_data,
    train_data,
    val_data,
    test_data,
    new_node_val_data,
    new_node_test_data,
) = get_link_prediction_data(
    dataset_name="WikiVote",
    val_ratio=0.15,
    test_ratio=0.15,
)

train_pos = (train_data.node_interact_sign == 1).sum()
test_pos = (test_data.node_interact_sign == 1).sum()
val_pos = (val_data.node_interact_sign == 1).sum()

train_rate = train_pos/len( train_data.node_interact_sign)
test_rate = test_pos/len(test_data.node_interact_sign)
val_rate = val_pos / len(val_data.node_interact_sign)

print(train_rate,test_rate,val_rate)