import pandas as pd
import re
import helper_function
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import functools as ft

reading_text_file=0
read_csv_file=1
#######For reading the text files we can use following code########################


def logs_to_df(logfile, output_dir, errors_file):
    with open(logfile) as source_file:
        linenumber = 0
        parsed_lines = []
        for line in tqdm(source_file):
            try:
                log_line = re.findall(combined_regex, line)[0]
                parsed_lines.append(log_line)
            except Exception as e:
                with open(errors_file, 'at') as errfile:
                    print((line, str(e)), file=errfile)
                continue
            linenumber += 1
            if linenumber % 250_00 == 0:
                df = pd.DataFrame(parsed_lines, columns=columns)
                df.to_parquet(f'{output_dir}/file_{linenumber}.parquet')
                parsed_lines.clear()
        else:
            df = pd.DataFrame(parsed_lines, columns=columns)
            df.to_parquet(f'{output_dir}/file_{linenumber}.parquet')
            parsed_lines.clear()


if reading_text_file==1:
    common_regex = '^(?P<client>\S+) \S+ (?P<userid>\S+) \[(?P<datetime>[^\]]+)\] "(?P<method>[A-Z]+) (?P<request>[^ "]+)? HTTP/[0-9.]+" (?P<status>[0-9]{3}) (?P<size>[0-9]+|-)'
    combined_regex = '^(?P<client>\S+) \S+ (?P<userid>\S+) \[(?P<datetime>[^\]]+)\] "(?P<method>[A-Z]+) (?P<request>[^ "]+)? HTTP/[0-9.]+" (?P<status>[0-9]{3}) (?P<size>[0-9]+|-) "(?P<referrer>[^"]*)" "(?P<useragent>[^"]*)'
    columns = ['client', 'userid', 'datetime', 'method', 'request', 'status', 'size', 'referer', 'user_agent']

    logs_to_df(logfile='access.log', output_dir='parquet_dir/',
               errors_file='errors.txt')
    logs_df = pd.read_parquet('parquet_dir/')

    logs_df['client'] = logs_df['client'].astype('category')
    del logs_df['userid']
    logs_df['datetime'] = pd.to_datetime(logs_df['datetime'], format='%d/%b/%Y:%H:%M:%S %z')
    logs_df['method'] = logs_df['method'].astype('category')
    logs_df['status'] = logs_df['status'].astype('int16')
    logs_df['size'] = logs_df['size'].astype('int32')
    logs_df['referer'] = logs_df['referer'].astype('category')
    logs_df['user_agent'] = logs_df['user_agent'].astype('category')

    logs_df.to_parquet('logs_df.parquet')
    logs_df = pd.read_parquet('logs_df.parquet')
    data=logs_df[['client','datetime','referer']]
    data.rename({'client':'IP','datetime':'Time','referer':'URL'},inplace=True)
    #data.head()


if read_csv_file==1:
    print('reading the csv file')
    data=pd.read_csv('sample_guardian_data.csv')
    ####Cleaning the dataset######
    data=data[data.IP.str.contains(r'[@#&$%+-/*]')]  ####Considering only those entries where we are finding '.'(part of IP address)

############Extraction of Characteristic Features##################

####getting the user list for creation of the profile#########
    #data['Time']=helper_function.cleaning_time(data['Time'])
    if (reading_text_file==0 or read_csv_file==1):
        data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S',errors='coerce')

    #data=helper_function.feeding_external_url_in_data(data) ####adding external links in the dataset to make it more usable
    #data.to_csv('updated_dataset.csv')
    top_out,tpvage=helper_function.cal_top(data)
    top_out.to_csv('time_on_page.csv')
    tos_out,tosave=helper_function.cal_tos(data)
    tos_out.to_csv('time_on_site.csv')
    #atp_out=helper_function.cal_atp(data)
    br_out=helper_function.cal_br(data)
    br_out.to_csv('bounce_rate.csv')
    er_out=helper_function.cal_er(data)
    er_out.to_csv('exit_rate.csv')
    cr_out=helper_function.cal_cr(data)
    cr_out.to_csv('cr_out.csv')
    nov_out=helper_function.cal_nov(data)
    nov_out.to_csv('nov_out.csv')
    apr_out=helper_function.avg_page_rank(data)
    apr_out.to_csv('apr_out.csv')
    chara_list=[top_out,br_out,er_out,cr_out,nov_out,apr_out]   ####For testing puporse we can use these 6 features as of now
    df_chara_list=ft.reduce(lambda left, right: pd.merge(left, right, on=['USER','URL'],how='inner'), chara_list)
    df_chara_list.to_csv('character_based_features.csv')

    results_knn = helper_function.apply_knn(df_chara_list)  ####Supervised Method with IPs as depenedent classes
    X_train, X_test, y_train, y_test=helper_function.feature_reduction_pca(df_chara_list)  ####Selecting the features from  PCA

    results_knn_pca=helper_function.apply_knn_pca(X_train, X_test, y_train, y_test)

    ######Another way of reducing the feature#########
    #helper_function.corr_analysis(df_chara_list)  ####This code will generate the heatmap
    results_dbscan = helper_function.apply_dbscan(df_chara_list)

#######Calculaiton of Content based features#########################
    #fk_out=helper_function.cal_fk(data)
    #fk_out.to_csv('frequency_kw.csv')