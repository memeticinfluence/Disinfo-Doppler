/*
  Slidemenu
*/
(function() {
    var $body = document.body;
    var $menu_trigger = $body.getElementsByClassName("menu-trigger")[0];

    if (typeof $menu_trigger !== "undefined") {
        $menu_trigger.addEventListener("click", function() {
            $body.className =
                $body.className == "menu-active" ? "" : "menu-active";
        });
    }
}).call(this);

/* Grid */


const { useState } = React;



var playing = false;
var interval;

function startplaying() {
    var btn = $("#btn-2");
    console.log(playing);
    if (playing) {
        playing = !playing;
        clearInterval(interval);
        btn.html(`<i class="fa fa-play"></i>`);
    } else {
        playing = !playing;
        $("#current-date").html(document.getElementById("start_date").value)
        get_mosaic(1)
        interval = setInterval(get_mosaic, 15000, 1);
        btn.html(`<i class="fa fa-pause"></i>`);
    }

}

function get_mosaic(offset) {
    document.getElementById("btn-1").disabled = true;
    document.getElementById("btn-2").disabled = true;
    document.getElementById("btn-3").disabled = true;
    document.getElementById("subreddits").disabled = true;
    var dt = document.getElementById("current-date").textContent;
    var subreddit = document.getElementById("subreddits").value;
    $("#app").html('<div class="text-center"><div class="spinner-border m-50" role="status" style="width: 15rem; height: 15rem;"><span class="sr-only">Loading...</span></div></div>')
    $("#current-date").html('<div class="text-center"><div class="spinner-border m-0" role="status" style="width: 3rem; height: 3rem;"><span class="sr-only">Loading...</span></div></div>')
    $.ajax({
        type: "GET",
        url: "/get_mosaic",
        data: {
            "subreddit": subreddit,
            "date": dt,
            "offset": offset,
            'playing': playing,
            'max_dt': document.getElementById("end_date").value,
            'min_dt': document.getElementById("start_date").value
        }
    }).then(function(response) {
        $("#app").html(response.body)
        $("#current-date").html(response.dt)
        document.getElementById("btn-1").disabled = false;
        document.getElementById("btn-2").disabled = false;
        document.getElementById("btn-3").disabled = false;
        document.getElementById("subreddits").disabled = false;
        $('#subreddits').selectpicker('refresh');
        document.getElementById("img_dwn").href = `/static/mosaics/${subreddit}_${response.dt.replaceAll("-", "")}.png`
    });
}
get_mosaic(0)

/*-----Developed by------|
|------Mitch Chaiet------|
|---memetic influence----|
|-------March 2021-------|
|-HKS Shorenstein Center-|

───────▄▀▀▀▀▀▀▀▀▀▀▄▄
────▄▀▀─────────────▀▄
──▄▀──────────────────▀▄
──█─────────────────────▀▄
─▐▌────────▄▄▄▄▄▄▄───────▐▌
─█───────────▄▄▄▄──▀▀▀▀▀──█
▐▌───────▀▀▀▀─────▀▀▀▀▀───▐▌
█─────────▄▄▀▀▀▀▀────▀▀▀▀▄─█
█────────────────▀───▐─────▐▌
▐▌─────────▐▀▀██▄──────▄▄▄─▐▌
─█───────────▀▀▀──────▀▀██──█
─▐▌────▄─────────────▌──────█
──▐▌──▐──────────────▀▄─────█
───█───▌────────▐▀────▄▀───▐▌
───▐▌──▀▄────────▀─▀─▀▀───▄▀
───▐▌──▐▀▄────────────────█
───▐▌───▌─▀▄────▀▀▀▀▀▀───█
───█───▀────▀▄──────────▄▀
──▐▌──────────▀▄──────▄▀
─▄▀───▄▀────────▀▀▀▀█▀
▀───▄▀──────────▀───▀▀▀▀▄▄▄*/