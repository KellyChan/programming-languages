$(document).ready(function(){

    $("p").css("background-color", "grey");

    $("#hide").click(function(){
        $(this).hide(1000, function(){
            alert("The paragraph is now hidden");
        });
    });

    $("#show").click(function(){
        $(this).show();
    });

    $("button").click(function(){
        $("#toggle").toggle();
    });

    $(".chaining").css("color", "red").click(function(){
        $(this).slideUp(2000)
            .slideDown(2000);
    });

});
