$(function (){
   $('#fileupload').fileupload({

       dataType: 'json',

       add: function (e, data){
           data.context = $('<p/>').text('Uploading...').appendTo(document.body);
           data.submit();
       },

       done: function(e, data){
           data.context.text('Upload finished.');
       }
   });
});


