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
    return render(request, 'main.html')

def get_mosaic(request):
    if request.is_ajax():
        subreddit = request.GET["subreddit"]
        date = pd.to_datetime(request.GET["date"])
        offset = request.GET["offset"]
        
        date += timedelta(days=int(offset))
        db = get_sql_connection()
        df = pd.read_sql(f'select * from mosaics where subreddit="{subreddit}" and dt="{date.strftime("%y%m%d")}"', db)
        df['idx'] = df['x'] + (df['y'] * (df['x'].max()+1))

        df = df.sort_values('idx')
        df['url'] = df['url'].apply(lambda x :x.replace('&amp;', '&'))
        gridbody = '<div class="grid">'

        for _, row in df.iterrows():
            gridbody += f'<a href="{row["full_link"]}" data-index="{row["idx"]}" class="item"><img style="width:100%; height:64" src="{row["url"]}"></img></a>'
        gridbody += "</div>"
        db.dispose()
        
        return JsonResponse({"body":gridbody, "dt":pd.to_datetime(date).date().isoformat()})
