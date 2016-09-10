$.getScript("http://www.parsecdn.com/js/parse-1.1.14.min.js", function() {
    $(document).ready(function() {
        $.getScript("https://raw.github.com/Mattieuga/test/master/internal_anon_user.js");
        $.getScript("https://raw.github.com/Mattieuga/test/master/internal_parse_subclasses.js");
        $.getScript("https://raw.github.com/Mattieuga/test/master/build_html.js", function() {
            $(window).trigger("html_loaded");
        });
    });
});
