$(document).ready(myClick);

function myAdd(){
    var toAdd = $('input[name=checkListItem]').val();
    $('.list').append('<div class="item">' + toAdd + '</div>');
}

function myClick(){
    $('#button').click(myAdd);
    $(document).on('click', '.item', myRemove);
}

function myRemove(){
    $(this).remove();
}
