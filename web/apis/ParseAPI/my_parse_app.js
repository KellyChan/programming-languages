$(window).on("html_loaded", function(e) {
    $(".submit_button").click(function(e) {
        
        // This creates our new Post object and 
        // sets some data
        var Post = Parse.Object.extend("Post");
        var myPost = new Post();        
        myPost.set("text", "Hello World!");
        
        // 1. Let's save this object to Parse using 
        //    the 'save()' function!
        myPost.save();
        
        
    });
});
