import pandas as pd
from requests_html import HTMLSession
from rake_nltk import Rake
from sklearn.cluster import DBSCAN
import random
from sklearn.decomposition import PCA
import collections
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize

def cleaning_time(x):
    new_time=[]
    x=x.tolist()
    for i in x:
        new_time.append(i[1:])
    return(new_time)


def cal_top(data,max_thr=150,min_thr=10):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    top_final=[]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        user_data['time_shift'] = user_data['Time'].shift(1)
        #print('The time difference is =====>', user_data['Time'] - user_data['time_shift'])
        user_data['time_spent'] = user_data['Time'] - user_data['time_shift']
        user_data['time_spent'] = user_data['time_spent'].apply(lambda x: x.total_seconds())
        user_data.groupby(['URL'])['time_spent'].sum()
        url_df=pd.DataFrame(user_data)
        url_df['USER']=user
        top_final.append(url_df)
    top=pd.concat(top_final)
    top=top[['USER','URL','time_spent']]
    print('===============>Final top')
    print(top)
    top_avg=top['time_spent'].sum()/len(user_list)
    top=top.groupby(['USER','URL'])['time_spent'].sum()
    top=top.reset_index()
    top.set_index('USER',inplace=True)
    return(top,top_avg)


def longest_string(x):
    main_site=x[:27]
    remaining=x[28:]
    content=remaining.split('/')[0]
    max_len=main_site+'/'+content
    return(max_len)

def cal_tos(data):
    final_tos = []
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        user_data['main_web_page'] = user_data['URL'].apply(longest_string)
        # unique_url=user_data['main_web_page'].unique()
        user_data['time_shift'] = user_data['Time'].shift(1)
        #print('The time difference is =====>', data['time_shift'] - data['Time'])
        user_data['time_spent'] = user_data['Time'] - user_data['time_shift']
        user_data['time_spent'] = user_data['time_spent'].apply(lambda x: x.total_seconds())
        user_data = user_data[user_data['time_spent'].notna()]
        user_data['time_spent']=user_data['time_spent'].astype(int)
        #####groupby will give us the aggregate timespent on each URL######
        tos = user_data.groupby(['main_web_page'])['time_spent'].sum()
        #tos.reset_index(inplace=True)
        tos_df=pd.DataFrame()
        tos_df['URL'] = tos.index
        tos_df['USER'] = str(user)
        tos_df['time_spent_on_site'] = tos.tolist()
        final_tos.append(tos_df)
    tos_cal = pd.concat(final_tos)
    tos_cal=tos_cal.loc[tos_cal['URL']!='HTTP']   ####Cleaning the simple HTTP Site
    tos_avg = tos_cal['time_spent_on_site'].sum() / len(user_data.index.unique())
    print(tos_cal)
    tos_cal = tos_cal.groupby(['USER', 'URL'])['time_spent_on_site'].sum()
    tos_cal = tos_cal.reset_index()
    tos_cal.set_index('USER',inplace=True)
    print('Final_TOS=============>')
    print(tos_cal)
    return(tos_cal,tos_avg)


def cal_atp(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        user_data['time_shift'] = user_data['Time'].shift(1)
        user_data['time_spent'] = user_data['Time'] - user_data['time_shift']
        user_data['time_spent'] = user_data['time_spent'].apply(lambda x: x.total_seconds())
        user_data = user_data[user_data['time_spent'].notna()]
        user_data['time_spent'] = user_data['time_spent'].astype(int)
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        atp_list=[]
        for url in user_data['URL'].tolist():
            atp_sum=0
            url_count=0
            for samples in minute_samples_10:
                if url in samples['URL'].tolist():
                    atp_sum=atp_sum+samples.loc[samples['URL']==url]['time_spent'].sum()
                    url_count=url_count+1
            if url_count>0 and atp_sum>0:
                atp_list.append([user,url,atp_sum/url_count])
            else:
                pass
        atp_df=pd.DataFrame(atp_list,columns=['USER','URL','AVG_TIME'])
    final_atp=pd.concat(atp_df)
    final_atp=final_atp.groupby(['USER','URL'])['AVG_TIME'].sum()
    final_atp=final_atp.reset_index()
    final_atp.set_index('USER',inplace=True)
    return(final_atp)

def cal_br(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    br_list=[]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        for session in range(len(minute_samples_10)):
            ops_df=minute_samples_10[session]
            count_hit=ops_df.groupby(['URL'])['IP'].count().values[0]*100/len(ops_df)
            #print(count_hit)
            ops_df['count_percentage']=count_hit
            ops_df['session_no']=session
            br_list.append(ops_df)
    final_br=pd.concat(br_list)
    final_br=final_br[['IP','URL','session_no','count_percentage']]
    final_br.rename(columns={'IP':'USER'},inplace=True)
    final_br=final_br.groupby(['USER','URL'])['count_percentage'].sum()
    print('Final BR===========================>')
    final_br=final_br.reset_index()
    final_br.set_index('USER',inplace=True)
    print(final_br)
    return (final_br)


def cal_er(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    url_count=[]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        url_list = user_data['URL'].unique()
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        for url in url_list:
            count=0
            for session in minute_samples_10:
                if url in session['URL'].values[-1]:  ####Checking specifically that do we have this URL in the last accessed web page or not
                    count=count+1
            if (count>0):
                url_count.append([user,url,count*100*len(minute_samples_10)/count])
    final_er=pd.DataFrame(url_count,columns=['USER','URL','Exit_Rate'])
    final_er=final_er.groupby(['USER','URL'])['Exit_Rate'].sum()
    final_er=final_er.reset_index()
    final_er.set_index('USER',inplace=True)
    print('final_ER======>',final_er)
    return(final_er)

def cal_cr(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    url_count = []
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        url_list=user_data['URL'].unique()
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        for url in url_list:
            count=0
            for session in minute_samples_10:
                if url in session['URL'].values:
                    count=count+1
                    url_count.append([user,url,count*100/len(minute_samples_10)])
    final_cr=pd.DataFrame(url_count,columns=['USER','URL','Conversion Rate'])
    final_cr=final_cr.groupby(['USER','URL'])['Conversion Rate'].sum()
    final_cr=final_cr.reset_index()
    final_cr.set_index('USER',inplace=True)
    print('The summary for conversion rate is========>')
    print(final_cr)
    return(final_cr)

def cal_nov(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    url_list=data['URL'].unique()
    nov_list=[]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        #url_list = user_data['URL'].unique()
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        for url in url_list:
            count=0
            for session in minute_samples_10:
                if url in session['URL'].values:
                    count = count + 1
            nov_list.append([user,url,count])   #####Standarizing the sum of count , becasue we might have different number of sessions for differnt users
    final_nov=pd.DataFrame(nov_list,columns=['USER','URL','NOV'])
    final_nov=final_nov.groupby(['USER','URL'])['NOV'].sum()
    final_nov=final_nov.reset_index()
    final_nov.set_index('USER',inplace=True)
    print('Final NOV =================>')
    print(final_nov)
    return (final_nov)


def cal_page_wgt(user_data):
    ####preprocessing on user df####
    pw_df=pd.DataFrame()
    user_name=user_data['IP'].unique()[0]
    user_data['time_shift'] = user_data['Time'].shift(1)
    user_data['time_spent'] = user_data['Time'] - user_data['time_shift']
    user_data['time_spent'] = user_data['time_spent'].apply(lambda x: x.total_seconds())
    user_data = user_data[user_data['time_spent'].notna()]
    user_data['time_spent'] = user_data['time_spent'].astype(int)
    nov_data = user_data.groupby(['URL'])['URL'].count()  ####This would be the NOV
    tp_spend=user_data.groupby(['URL'])['time_spent'].sum()
    #print('Time spend for pw is =========>')
    pw_df.index=tp_spend.index
    pw_df['NOV']=nov_data.values
    pw_df['time_spend']=tp_spend.values
    pw_df['pw']=pw_df['NOV']*pw_df['time_spend']
    pw_df['USER']=user_name
    #print('Nov for pw is==========>')
    #print(nov_data*tp_spend)
    pw_df['page_rank']=pw_df['time_spend'].rank(ascending=0)
    return(pw_df)

def avg_page_rank(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    final_cal = []
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        pg_wt=cal_page_wgt(user_data)
        damping_factor=0.85 #####constant damping factor
        url_list = user_data['URL'].unique()
        minute_samples_10 = [df for i, df in user_data.groupby(pd.Grouper(key='Time', freq="10min"))]
        minute_samples_10 = [ele for ele in minute_samples_10 if len(ele) != 0]
        for url in url_list:
            N=0
            pw_sum=0
            pr_sum=0
            for session in minute_samples_10:
                if url in session['URL'].values:
                    N=N+1
            pw=pg_wt.loc[pg_wt.index==url]['pw'].values[0]   ####getting the pw for calculation of APR
            pr=pg_wt.loc[pg_wt.index==url]['page_rank'].values[0] ######calculation of page rank is not mentioned in the paper so we are assuming this logic for page rank
            #######Calculation of ARP###########
            pw_sum=pw_sum+pw
            pr_sum=pr_sum+pr
            arp=1/N *(0.15+ 0.15*(pw_sum*pr_sum)/N)
            final_cal.append([user,url,arp])
    arp_total=pd.DataFrame(final_cal,columns=['USER','URL','Average_page_rank'])
    arp_total=arp_total.groupby(['USER','URL'])['Average_page_rank'].sum()
    arp_total=arp_total.reset_index()
    arp_total.set_index('USER',inplace=True)
    print('Final Average rank page is =============>')
    print(arp_total)
    return (arp_total)

#################Calculating content based features#####################


def extract_text(links):
    s = HTMLSession()
    response = s.get(links)
    return response.html.find('div#maincontent', first=True).text

def feeding_external_url_in_data(data):
    external_url_list = [
        'https://www.theguardian.com/commentisfree/2022/aug/18/there-is-very-little-to-celebrate-about-these-a-level-results-inequalities-just-got-worse'
        ,'https://www.theguardian.com/commentisfree/2022/aug/18/china-convenient-historical-tales-taiwan'
        ,'https://www.theguardian.com/football/2022/aug/18/deloitte-report-premier-league-leads-european-footballs-financial-recovery'
        ,'https://www.theguardian.com/us-news/2022/aug/18/texas-police-christopher-shaw-civil-rights'
        ,'https://www.theguardian.com/commentisfree/2022/aug/19/bp-energy-sector-incompetent-ofgem-freeze-price-cap-tax-profits'
        ,'https://www.theguardian.com/sport/2022/aug/19/dina-asher-smith-praised-for-shattering-massive-taboo-around-periods-in-sport'
        ,'https://www.theguardian.com/sport/2022/aug/19/emma-raducanus-run-in-cincinnati-ends-with-defeat-to-jessica-pegula'
        ,'https://www.theguardian.com/sport/live/2022/aug/19/england-v-south-africa-first-test-day-three-live-score-updates-cricket'
        ,'https://www.theguardian.com/us-news/2022/aug/19/andrew-tate-instagram-facebook-removed'
        ,'https://www.theguardian.com/music/2022/aug/19/it-was-sacrilegious-why-the-destruction-of-manchesters-ian-curtis-mural-struck-a-nerve'
        ,'https://www.theguardian.com/games/2022/aug/19/in-the-callisto-protocol-dark-and-terrible-things-lurk-in-space']
    random_index=random.sample(range(len(data)), len(data))
    print(random_index)
    for i in random_index:
        index_df=data.index[i]
        data.loc[index_df,'URL']=random.choice(external_url_list)
    return(data)

def removing_stopwords(data):
    word_tokens = word_tokenize(data)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return(filtered_sentence)

def find_kw(data):
    #whole_text=[]
    unique_urls=data['URL'].unique()
    for url in unique_urls:
        f = extract_text(url)
        whole_text = removing_stopwords(f)
        whole_text = " ".join([str(item) for item in whole_text])
        #print('removed all the english stopwords')
        if len(whole_text)>50:
            r=Rake()
            r.extract_keywords_from_text(whole_text)
            result=r.get_ranked_phrases_with_scores()
            df_kw=pd.DataFrame(result,columns=['frequency','keyword'])
            df_kw['USER']=data['IP'].unique()[0]
    #print('keywords for user',data['IP'].unique()[0],'is=====>',df_kw)
            return(df_kw)
        else:
            df_kw=pd.DataFrame()    #####returning empty df
            return (df_kw)



def cal_fk(data):
    user_list = data.groupby(['IP']).count().index.tolist()[:-1]
    final_kw=[]
    for user in user_list:
        user_data = data.loc[data['IP'] == user]
        df=find_kw(user_data)
        if len(df)>0:
            df.sort_values('frequency')
            final_kw.append(df[:int(0.75*len(df))])
    final_kw=pd.concat(final_kw)
    print('The keywords are as follows=======>')
    print(final_kw)
    return(final_kw)

def apply_knn_pca(X_train, X_test, y_train, y_test):
    K = []
    training = []
    test = []
    scores = {}
    print('The results on Knn for selected features using PCA is===============>')
    for k in range(2, 21):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        training_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        K.append(k)
        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]

    for keys, values in scores.items():
        print(keys, ':', values)
    return (scores)




def apply_knn(all_combined_data):
    print(all_combined_data)
    y=all_combined_data.index
    X=all_combined_data[['time_spent','count_percentage','Exit_Rate','Conversion Rate','NOV','Average_page_rank']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    K = []
    training = []
    test = []
    scores = {}

    for k in range(2, 21):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        training_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        K.append(k)
        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]

    for keys, values in scores.items():
        print(keys, ':', values)
    return(scores)


def corr_analysis(df_chara_list):
    cor=df_chara_list.corr()
    sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns)
    s = cor.unstack()
    so = s.sort_values(kind="quicksort")
    print(so)

def feature_reduction_pca(all_combined_data):
    y = all_combined_data.index
    X = all_combined_data[['time_spent', 'count_percentage', 'Exit_Rate', 'Conversion Rate', 'NOV', 'Average_page_rank']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) 
    explained_variance = pca.explained_variance_ratio_
    print('The explained variance by PCA is=========>',explained_variance)
    return(X_train, X_test, y_train, y_test)

def apply_dbscan(all_combined_data):
    X = all_combined_data[['time_spent', 'count_percentage', 'Exit_Rate', 'Conversion Rate', 'NOV', 'Average_page_rank']]
    clustering = DBSCAN(eps=5, min_samples=10).fit(X)
    number_of_clusters=clustering.labels_
    counter = collections.Counter(number_of_clusters)
    print('DBSCAN has following clusters for the input dataset')
    print(counter)
    return(counter)