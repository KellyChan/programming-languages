$(function(){
    $('#fileupload').fileupload({

        url: 'url',
        dataType: 'json',
        disableImageResize: /Android(? !.*Chrome)|Opera/
            .test(window.navigator && navigator.userAgent),
        imageMaxWidth: 800,
        imageMaxHeight: 800,
        imageCrop: true
    });
});
