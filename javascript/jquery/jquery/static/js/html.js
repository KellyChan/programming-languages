$(document).ready(function(){

    $("#btn1").click(function(){
        alert("Text: " + $("#test").text());
    });

    $("#btn2").click(function(){
        alert("HTML: " + $("#test").html());
    });

    $("#btn3").click(function(){
        alert("Value: " + $("#test").val());
    });



});