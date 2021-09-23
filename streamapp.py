from urllib.parse import uses_relative
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('ipl.csv')
df['year'] = pd.to_datetime(df.date).dt.year
df = df[df.year>2016]

st.sidebar.header('IPL statistical analysis')
side_choice = st.sidebar.selectbox('Select', ['Statistical Predictions','Batting','Bowling','Teams','Venue'])



if side_choice == 'Statistical Predictions':
    form = st.form(key='my-form')
    venue = form.selectbox('Select venue', df.venue.unique(), key=1)
    team1 = form.selectbox('Select Team', df.team1.unique(), key=2)
    team2 = form.selectbox('vs', df.team2.unique(), key=3)
    submit = form.form_submit_button('Predict')
    
    df['year'] = pd.to_datetime(df.date).dt.year
    df = df[df.year>2016]
    df1 = df[['venue','team1','team2','winner']]
    df1.fillna(0,inplace=True)
    X= pd.get_dummies(df1[['venue','team1','team2']],drop_first=True)
    y = df1[['winner']]
    df1.fillna(0,inplace=True)

    eg = df1.winner.value_counts()
    win = pd.DataFrame(eg).index
    winners=[]
    for i  in win:
        winners.append(i)


    def rank_team(x):
        for index,i in enumerate(winners):
            if x==i:
                return index
    df1['winner'] = df1.winner.apply(lambda x: rank_team(x))
    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb = MultiLabelBinarizer()


    X= pd.get_dummies(df1[['venue','team1','team2']])
    # y = mlb.fit_transform(df1[['winner']])
    y = df1.winner

    model = LogisticRegression()

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = None)


    model.fit(X_train,y_train)
    # from sklearn.metrics import confusion_matrix
    # print(confusion_matrix(y_test,pred))

    # (accuracy_score(y_test,pred)*100)



    if submit:

        # from sklearn.metrics import confusion_matrix
        # print(confusion_matrix(y_test,pred))
        df4=df1.drop('winner',axis=1)
        input = df4.iloc[0]
        input['venue']=venue
        input['team1']=team1
        input['team2']=team2
        
        too_pred = X_train.iloc[0]
        to_pred = X_train.iloc[0]
        for index,i in enumerate(too_pred):
            to_pred[index]=0
        
        to_pred[f"""venue_{input['venue']}"""]=1
        to_pred[f"""team1_{input['team1']}"""]=1
        to_pred[f"""team2_{input['team2']}"""]=1

        pred=np.array(to_pred).reshape(1,-1)

        for index,winn in enumerate(winners):
            if index==model.predict(pred)[0]:
                st.header(winn)
                break


elif side_choice == 'Batting':
    

    st.title('IPL Batting statistics')

    ipl = pd.read_csv('ipl.csv')
    ipl_ball = pd.read_csv('iplball.csv')




    st.header('Dataset')
    st.markdown('Data grouped by batsman and summarized by batting statistical features')




    ipl_bat_table = pd.read_csv('ipl_bat_table.csv').iloc[:,1:].fillna(0)
    st.dataframe(ipl_bat_table)
    batting_runs = pd.read_csv('batting_runs.csv')


    
    st.header('Plots')

    st.subheader('Runs vs balls')
    df = batting_runs.iloc[:,1:].fillna(0)
    fig = px.scatter(df, x="Balls_Faced", y="Runs_scored", text="batsman", log_x=True, size_max=100,color="Matches_played")
    fig.update_traces(textposition='top center')
    fig.update_layout(title_text='Total runs vs total balls faced', title_x=0.5)
    st.plotly_chart(fig,use_container_width=True)


    st.subheader('Hundreds vs Fifties')
    bat = ipl_bat_table
    x = st.slider('x',15,bat.shape[0]-15)  # 👈 this is a widget
    upto=range(0,x+15)
    t = ipl_bat_table[['batsman','Hundreds',"Fifties"]].iloc[upto].sort_values(['Fifties','Hundreds'])
    h = t[['batsman','Hundreds']]
    h['type'] = 'Hundred'
    h.rename(columns = {'Hundreds':'Value'},inplace=True)
    f = t[['batsman','Fifties']]
    f['type'] = 'Fifties'
    f.rename(columns = {'Fifties':'Value'},inplace=True)
    ghy = pd.concat([h,f]).sort_values('Value',ascending=False)


    fig = px.histogram(ghy, x="batsman", y="Value", color="type",
                        hover_data=ghy.columns)

    # Plot!
    st.plotly_chart(fig, use_container_width=True)


    st.subheader('Fours vs Sixes')

    y = st.slider('y',15,bat.shape[0]-15)  # 👈 this is a widget
    upto=range(0,y+15)


    def bar_data(bat,i='Fours',j='Sixes'):
        t = bat[['batsman',i,j]].iloc[upto].sort_values([i,j],ascending=False)
        h = t[['batsman',i]]
        
            
            
        h['type'] = i
        h.rename(columns = {i:'Value'},inplace=True)
        f = t[['batsman',j]]
        f['type'] = j
        f.rename(columns = {j:'Value'},inplace=True)
        ghy = pd.concat([h,f]).sort_values('Value',ascending=False)
        return ghy
    ghy = bar_data(bat,'Fours','Sixes')
    fig = px.histogram(ghy, x="batsman", y="Value", color="type",
                        hover_data=ghy.columns)

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

elif side_choice == 'Bowling':


    st.title('IPL Bowling statistics')

    ipl = pd.read_csv('ipl.csv')
    iplball = pd.read_csv('iplball.csv')




    st.header('Dataset')
    st.markdown('Data grouped by bowler and summarized by bowling statistical features')




    bowling = pd.read_csv('bowling.csv').iloc[:,1:].fillna(0)
    st.dataframe(bowling)
    


    
    st.header('Plots')

    st.subheader('Wickets vs Matches')
    st.markdown('Color and size represent total balls bowled and runs conceded respectively')
    min_wik = st.slider('Minimum total wickets to show',0.0,bowling.total_wickets.max())
    fig = px.scatter(bowling[bowling['total_wickets']>=min_wik], x="matches", y="total_wickets",
	         size="total_runs", color="total_balls_bowled",
                 hover_name="bowler", log_x=True, size_max=40)
    st.plotly_chart(fig, use_container_width=True,unsafe_allow_html=True)


    st.subheader('Strike rate vs Average vs Economy')
    
    no_of_players_to_show = st.slider('No of players to show',0,bowling.shape[0]-15)  # 👈 this is a widget
    upto=range(0,no_of_players_to_show+15)
    bowl_plot = bowling[['bowler','average','strike_rate','economy']].iloc[upto]
    bowl_plot1 = bowl_plot.melt(id_vars='bowler',value_vars=['strike_rate','average','economy'])
    fig = px.histogram(bowl_plot1, x="bowler", y="value",color='variable')

    # Plot!
    st.plotly_chart(fig, use_container_width=True)


    st.subheader('Dot balls vs fours and sixes conceded')


    c = st.slider('No of players to show',15,bowling.shape[0]-15)  # 👈 this is a widget
    upto=range(0,c+15)
    over_balls = iplball.groupby(['bowler','id','inning','over','ball'])[['total_runs','extra_runs']].sum().reset_index()
    fours_con = over_balls[(over_balls['total_runs']-over_balls['extra_runs'])==4].groupby('bowler')['total_runs'].count().to_frame('fours').reset_index()
    sixes_con = over_balls[(over_balls['total_runs']-over_balls['extra_runs'])==6].groupby('bowler')['total_runs'].count().to_frame('sixes').reset_index()
    f_s = pd.merge(fours_con,sixes_con,how='left',left_on=['bowler'],right_on=['bowler'])
    f_s['sixes'] = f_s['sixes'].fillna(0).astype(int)
    bowl_plot = bowling[['bowler','total_balls_bowled','dot_balls','boundaries_4-6']]
    bowl_plot = pd.merge(bowl_plot,f_s,how='left',left_on=['bowler'],right_on=['bowler'])
    bowl_plot1 = bowl_plot.melt(id_vars='bowler',value_vars=['dot_balls','fours','sixes'])
    bowl_plot1 = pd.merge(bowl_plot1,bowling[['bowler','total_balls_bowled']],how='left',left_on=['bowler'],right_on=['bowler']).sort_values(['total_balls_bowled','value'],ascending=False)
    fig = px.histogram(bowl_plot1.iloc[upto], x="bowler", y="value",color='variable',hover_data=['total_balls_bowled'])
    # Plot!
    st.plotly_chart(fig, use_container_width=True)



elif side_choice == 'Teams' or side_choice == 'Venue':
    st.header('In development')


    # st.title('title')
    # st.markdown('_Markdown_') # see *
    # st.latex(r''' e^{i\pi} + 1 = 0 ''')
    # st.write('Most objects') # df, err, func, keras!
    # st.write(['st', 'is <', 3]) # see *
    # st.title('My title')
    # st.header('My header')
    # st.subheader('My sub')
    # st.code('for i in range(8): foo()')
    # # * optional kwarg unsafe_allow_html = True
    # st.caption('This is a small text')