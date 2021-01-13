import pandas as pd #for data analytics
from datetime import datetime, date, timedelta

# This program uses a few date manipulation techniques 
# These are verty useful in extracting several flavors of date

def get_daily_data():
    pivot_date = datetime.now().date()- pd.DateOffset( months = 1)
    day_t_present = pivot_date +timedelta(-pivot_date.day + 1)
    day_t_present_c = datetime.strftime(day_t_present, '%Y-%m-%d')
    day_tminus1_present = datetime.strftime(day_t_present - pd.DateOffset(months =1),'%Y-%m-%d')
    day_t_past = day_t_present - pd.DateOffset(years = 1)
    day_t_past_c = datetime.strftime(day_t_past, '%Y-%m-%d')
    day_tminus1_past = datetime.strftime(day_t_past - pd.DateOffset(months = 1), '%Y-%m-%d')
    day_tplus1_past = datetime.strftime(day_t_past +  pd.DateOffset(months=1), '%Y-%m-%d')
    return day_t_present_c, day_tminus1_present, day_t_past_c, day_tminus1_past, day_tplus1_past

def main():
    date = get_daily_data()
    print("Day1 a month ago from current date ->", date[0])
    print("Day1 two months ago from current date ->", date[1])
    print("Day1 a month ago of last year from current date ->", date[2])
    print("Day1 two months ago of last year from current date ->", date[3])
    print("Day1 a month ahead of last year from current date ->", date[4])
    
    
    
if __name__ == "__main__":
    main()
