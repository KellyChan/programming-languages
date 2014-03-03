# To experiment with this code freely you will have to run this code locally.
# We have provided an example json output here for you to look at,
# but you will not be able to run any queries through our UI.
import json
import requests


BASE_URL = "http://musicbrainz.org/ws/2/"
ARTIST_URL = BASE_URL + "artist/"

query_type = {  "simple": {},
                "atr": {"inc": "aliases+tags+ratings"},
                "aliases": {"inc": "aliases"},
                "releases": {"inc": "releases"}}


def query_site(url, params, uid="", fmt="json"):
    params["fmt"] = fmt
    r = requests.get(url + uid, params=params)
    print "requesting", r.url

    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        r.raise_for_status()


def query_by_name(url, params, name):
    params["query"] = "artist:" + name
    return query_site(url, params)


def pretty_print(data, indent=4):
    if type(data) == dict:
        print json.dumps(data, indent=indent, sort_keys=True)
    else:
        print data


def main():
    results = query_by_name(ARTIST_URL, query_type["simple"], "First Aid Kit")
    #pretty_print(results)

    artist_id = results["artist"][1]["id"]
    #print "\nARTIST:"
    #pretty_print(results["artist"][1])

    artist_data = query_site(ARTIST_URL, query_type["releases"], artist_id)
    releases = artist_data["releases"]
    #print "\nONE RELEASE:"
    #pretty_print(releases[0], indent=2)
    #release_titles = [r["title"] for r in releases]

    #print "\nALL TITLES:"
    #for t in release_titles:
    #    print t


    results = query_by_name(ARTIST_URL, query_type["simple"], "FIRST AID KIT")
    #pretty_print(results)
    counter = 0
    for artist in results['artist']:
        if artist['score'] == "100":
            counter += 1
    print "\n# of FIRST AID KIT:"
    print counter

    results = query_by_name(ARTIST_URL, query_type["simple"], "Queen")
    #pretty_print(results)
    begin_area = results['artist'][0]['begin-area']['name']
    print "\nbegin-area named for 'Queen':"
    print begin_area

    results = query_by_name(ARTIST_URL, query_type["simple"], "Beatles")
    #pretty_print(results)
    aliases = results['artist'][0]['aliases']
    for alias in aliases:
        if alias['locale'] == 'es':
            alias_name = alias['name']
    print "\nSpanish alise for Beatles:"
    print alias_name


    results = query_by_name(ARTIST_URL, query_type["simple"], "Nirvana")
    disambiguation = results['artist'][0]['disambiguation']
    print "\nNirvana Disambiguation:"
    print disambiguation


    results = query_by_name(ARTIST_URL, query_type["simple"], "One Direction")
    begin_date = results['artist'][0]['life-span']['begin']
    print "\nWhen was 'One Direction' formed:"
    print begin_date



if __name__ == '__main__':
    main()
