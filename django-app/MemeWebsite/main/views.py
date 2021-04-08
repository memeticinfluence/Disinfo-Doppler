from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta

def get_sql_connection():
    db = create_engine(
        "mysql://admin:65Qyw6pgSo8F3LyiASFr@meme-observatory.cizj1wczwqh5.us-west-2.rds.amazonaws.com:3306/meme_observatory?charset=utf8",
        encoding="utf8",
    )
    return db



def main(request):
    db = get_sql_connection()
    options = list(pd.read_sql('select distinct(subreddit) from mosaics order by subreddit', db).iloc[:, 0])
    db.dispose()

    return render(request, 'main.html', context={'options':options})

def get_mosaic(request):
    if request.is_ajax():
        subreddit = request.GET["subreddit"]
        date = pd.to_datetime(request.GET["date"])
        print(request.GET["playing"])
        
        offset = request.GET["offset"]
        
        date += timedelta(days=int(offset))
        
        max_date = pd.to_datetime(request.GET["max_dt"])
        min_date = pd.to_datetime(request.GET["min_dt"])

        if (bool(request.GET['playing'])) and date > max_date:
            date = min_date
        
        db = get_sql_connection()
        df = pd.read_sql(f'select * from mosaics where subreddit="{subreddit}" and dt="{date.strftime("%y%m%d")}"', db).drop_duplicates()
        df['idx'] = df['x'] + (df['y'] * (df['x'].max()+1))

        df = df.sort_values('idx')
        df['url'] = df['url'].apply(lambda x :x.replace('&amp;', '&'))
        gridbody = '<div class="grid">'

        for _, row in df.iterrows():
            gridbody += f'<a href="{row["full_link"]}" data-index="{row["idx"]}" class="item"><img style="width:100%; height:80" src="{row["url"]}"></img></a>'
        gridbody += "</div>"
        db.dispose()

        return JsonResponse({"body":gridbody, "dt":pd.to_datetime(date).date().isoformat()})
