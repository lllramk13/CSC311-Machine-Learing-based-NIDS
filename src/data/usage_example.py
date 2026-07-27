from src.data.loader import load_flows, training_views, assign_splits

# get the date
flows = load_flows(
    datasets=["unsw_nb15"],
    labels=[0, 1],
)

# 70% train、15% validation、15% test
flows = assign_splits(
    flows,
    train_percentage=70,
    validation_percentage=15,
)
train_flows = flows.filter("Split = 'train'")
validation_flows = flows.filter("Split = 'validation'")
test_flows = flows.filter("Split = 'test'")


X_train, y_train, train_metadata = training_views(train_flows)
X_val, y_val, val_metadata = training_views(validation_flows)
X_test, y_test, test_metadata = training_views(test_flows)