import requests
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
s.mount('http://', HTTPAdapter(max_retries=retries))

def build_api_endpoint(subreddit, size, ascending, last_record_date, verbose):
    '''
    An easy API endpoint builder for PushShift's Reddit API.
    '''
    url_base = ('http://api.pushshift.io/reddit/submission/search/'
               f'?subreddit={ subreddit }&size={ size }')
    if not last_record_date:
        url = url_base
    else:
        if ascending:
            url = url_base + f'&after={ last_record_date + 1 }&sort=asc'
        else:
            url = url_base + f'&before={ last_record_date - 1 }&sort=desc'
    if verbose:
        print(url)
        
    return url


def download_subreddit_posts(subreddit, size=5000, ascending=False, 
                             start_date=False, seen_ids=set(), 
                             display_every_x_iterations=20,
                             verbose=True):
    '''
    Queries the pushshift.io API for a given `subreddit`.
    
    To go back in time from `start_date`, set ascending to False.
    To go forward from time `state_date`, set ascneding to True.
    Skips ids in set `seen_ids`.
    Displays the http request every `display_every_x_iterations` if `verbose`=True.
    '''
    if not isinstance(seen_ids, set):
        raise "seen_ids needs to be a set!"
    i = 0
    records = []
    last_record_date = start_date
    try:
        while True:
            # Buld the url
            url = build_api_endpoint(subreddit, size, ascending, last_record_date, 
                                     verbose if i % display_every_x_iterations == 0 else False)

            # make the HTTP request to the API
            r = s.get(url)
            resp = r.json()
            data = resp.get('data')

            # check which records were returned by the API and which are new?
            # if there are no new IDs, then we're done!
            paginated_ids = {row.get("id") for row in data}
            new_ids = paginated_ids - seen_ids
            if len(new_ids) == 0:
                break

            # add new records to existing records. 
            new_records = [row for row in data if row['id'] in new_ids]
            records.extend(new_records)

            # collect all records created before the last record's date.
            last_record = data[-1]
            last_record_date = last_record.get('created_utc')
            i += 1
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        if verbose:
            print("Cancelled early")
        
    return records
