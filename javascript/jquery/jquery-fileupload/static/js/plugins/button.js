$(function(){
    $('#fileupload').fileupload({

        dataType: 'json',

        add: function (e, data){
            data.context = $('<button/>').text('Upload')
                                         .appendTo(document.body)
                                         .click(function(){
                                             data.context = $('<p/>').text('Uploading...').replaceAll($(this));
                                             data.submit();
                                          });
        },


        done: function (e, data) {
            data.context.text('Upload finished.');
        }
    });
});

