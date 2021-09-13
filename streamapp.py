from urllib.parse import uses_relative
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.sidebar.header('IPL statistical analysis')
side_choice = st.sidebar.selectbox('Select', ['Batting','Bowling','Teams','Venue'])

if side_choice == 'Batting':
    

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