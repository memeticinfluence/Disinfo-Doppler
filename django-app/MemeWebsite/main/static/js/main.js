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


$(function() {
    /**
     * Store the transition end event names for convenience.
     */
    var transitionEnd =
        "transitionend webkitTransitionEnd oTransitionEnd MSTransitionEnd";

    /**
     * Trigger the play button states upon clicking.
     */
    $(".play-btn").click(function(e) {
        e.preventDefault();

        if ($(this).hasClass("stop")) {
            $(this).removeClass("stop").addClass("to-play");
        } else if (!$(this).hasClass("to-play")) {
            $(this).addClass("stop");
        }
    });

    /**
     * Remove the 'to-play' class upon transition end.
     */
    $(document).on(transitionEnd, ".to-play", function() {
        $(this).removeClass("to-play");
    });
});

const { useState } = React;

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