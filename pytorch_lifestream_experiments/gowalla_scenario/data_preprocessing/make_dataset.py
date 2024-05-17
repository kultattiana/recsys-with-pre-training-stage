from ptls.make_datasets_spark import DatasetConverter

import pyspark.sql.functions as F
import logging
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import pyspark.pandas as ps
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexerModel
import pyspark.sql.types as T
import json
import os

FILE_NAME_TRAIN = 'gowalla_user_activity.csv'
FILE_NAME_TEST = 'test.csv'
COL_EVENT_TIME = 'TRDATETIME'

logger = logging.getLogger(__name__)


class LocalDatasetConverter(DatasetConverter):


    def load_transactions(self):
        df = self.spark_read_file(self.path_to_file(FILE_NAME_TRAIN))
        logger.info(f'Loaded {df.count()} records from "{FILE_NAME_TRAIN}"')
        
        #df = df.withColumn('event_time', F.to_timestamp('check-in time', "yyyy-MM-dd'T'HH:mm:ss'Z'"))
        df = df.withColumn('event_time', F.col('check-in time'))
        print(df.dtypes)

        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower())
        
        copy_df = df

        df = df.drop('check-in time')
        df.show()

        train_feature_df, train_target_df, test_feature_df, test_target_df = self.collect_lists(df, self.config.col_client_id)
        print("After collecting lists feature", train_feature_df.count())
        print("After collecting lists target", train_target_df.count())

        #feature_df = feature_df.filter(~(F.array_contains(feature_df.location_id, zero_key)))
        #target_df = target_df.filter(~(F.array_contains(target_df.location_id, zero_key)))

        train_target_users = [row.user for row in train_target_df.select('user').collect()]
        train_feature_users = [row.user for row in train_feature_df.select('user').collect()]
        test_target_users = [row.user for row in test_target_df.select('user').collect()]
        test_feature_users = [row.user for row in test_feature_df.select('user').collect()]
        train_feature_df = train_feature_df.filter(train_feature_df.user.isin(train_target_users))
        train_target_df = train_target_df.filter(train_target_df.user.isin(train_feature_users))
        test_feature_df = test_feature_df.filter(test_feature_df.user.isin(test_target_users))
        test_target_df = test_target_df.filter(test_target_df.user.isin(test_feature_users))

        print("Count after filter feature", train_feature_df.count())
        print("Count after filter target", train_target_df.count())
        print("Count after filter feature", test_feature_df.count())
        print("Count after filter target", test_target_df.count())
        
        return train_feature_df, train_target_df, test_feature_df, test_target_df
        
    
    def run(self):
        self.parse_args()
        self.logging_config()

        
        logger.info(f'Start processing')
        train_feature_df, train_target_df, test_feature_df, test_target_df = self.load_transactions()
        
        #self.process_train_partition(feature_df, target_df)
        train_path_to = self.config.output_train_path
        test_path_to = self.config.output_test_path

        train_feature_df.write.parquet(train_path_to, mode='overwrite')
        test_feature_df.write.parquet(test_path_to, mode='overwrite')

        train_target_df.write.parquet('data/target_train_trx.parquet', mode='overwrite')
        test_target_df.write.parquet('data/target_test_trx.parquet', mode='overwrite')
        train_feature_df.write.parquet('data/train_trx.parquet', mode='overwrite')
        test_feature_df.write.parquet('data/test_trx.parquet', mode='overwrite')
        

        logger.info(f'{FILE_NAME_TRAIN} - done. '
                    f'Train: {train_feature_df.count()}, test: {test_feature_df.count()}')
        logger.info(train_feature_df.dtypes)
        

       
        logger.info(f'Data collected')
    
    

    def collect_lists(self, df, col_id):
        #df = df.drop("_c0")

        def parse_list_from_string(x):
            res = json.loads(x)
            return res

        location_list = F.udf(parse_list_from_string, T.ArrayType(T.LongType()))
        df.show()
        df = df.withColumn("location_id_list", location_list(F.col("location_id_bin")))
        df = df.drop("location_id_bin").withColumnRenamed("location_id_list","location_id_bin")
        df = df.withColumn("event_time_list", location_list(F.col("event_time")))
        df = df.drop("event_time").withColumnRenamed("event_time_list","event_time")
        print(df.dtypes)

        col_list = [col for col in df.columns if (col != col_id) & (col != '_c0')]
        #df = df.withColumn('_rn', F.row_number().over(Window.partitionBy(col_id).orderBy('event_time')))
        train_feature_df = df

        train_feature_df = train_feature_df.withColumn('location_id_target', F.slice('location_id_bin', F.array_size('location_id_bin') - 1, 1).getItem(0).alias('location_id_bin'))

        #if feature_or_target == 'feature':
        for col in col_list:
            train_feature_df = train_feature_df.withColumn(col,
                F.slice(col, 1, F.array_size(col) - 2).alias(col))
            
        
        train_feature_df.show()
        
        train_target_df = df
            
        
        #if feature_or_target == 'target':
        for col in col_list:
      
            train_target_df = train_target_df.withColumn(col, 
                F.slice(col, F.array_size(col) - 1, 1).alias(col)
            )
            
        train_target_df.show()

        test_feature_df = df
        test_feature_df = test_feature_df.withColumn('location_id_target', F.slice('location_id_bin', F.array_size('location_id_bin'), 1).getItem(0).alias('location_id_bin'))
        
        for col in col_list:
            test_feature_df = test_feature_df.withColumn(col,
                F.slice(col, 2, F.array_size(col) - 2).alias(col))
            
    
        
        test_target_df = df
        test_feature_df.show()

        for col in col_list:
            test_target_df = test_target_df.withColumn(col,
                F.slice(col, F.array_size(col), 1).alias(col))
        
        test_target_df.show()
      
        return train_feature_df, train_target_df, test_feature_df, test_target_df


if __name__ == '__main__':
    LocalDatasetConverter().run()
