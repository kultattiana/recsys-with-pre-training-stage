export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[8] \
    --name "Brightkite Make Dataset" \
    --driver-memory 16G \
    --conf spark.sql.shuffle.partitions=60 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path 'data/' \
    --trx_files "#predefined" \
    --col_client_id "user" \
    --cols_log_norm "latitude" "longitude" \
    --cols_event_time "check-in time" \
    --target_files "#predefined" \
    --col_target "location_id" \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/brightkite_checkin_pred.txt"\
    --print_dataset_info
    #--cols_log_norm "check-in-time" \
    #--cols_category "latitude" "longitude" \