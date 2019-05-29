$(document).ready(function(){


    // upload single picture
	function readURL(input) {
	  if (input.files && input.files[0]) {
	    var reader = new FileReader();
	    reader.onload = function(e) {
	      $('#uploaded-img').css('background-image', 'url(' + e.target.result + ')');
	      $('#home').addClass('hide');
	      $('#process').removeClass('hide');
	    }
	    reader.readAsDataURL(input.files[0]);
	  }
	}

	$("#feature-upload-input").change(function() {
	  readURL(this);
	});

	$("#feature-upload-img, #feature-upload-txt").click(function () {
    	$("#feature-upload-input").trigger('click');
	});
	/***********/

    // upload multiple pictures
    $(function() {
    var imagesPreview = function(input, placeToInsertImagePreview) {
        if (input.files) {
            var filesAmount = input.files.length;
            for (i = 0; i < filesAmount; i++) {
                var reader = new FileReader();
                reader.onload = function(event) {
                    $($.parseHTML('<img>')).attr('src', event.target.result).appendTo(placeToInsertImagePreview);
                }
                reader.readAsDataURL(input.files[i]);
            }
        }
    };

    $('#feature-upload-multiple-input').on('change', function() {
        imagesPreview(this, 'div.advices');
    });

    $(".advices").click(function () {
    	$("#feature-upload-multiple-input").trigger('click');
	      $('.advices p').addClass('hide');
	      $('.advices #add-icon').addClass('hide');
	      $('.advices').css('padding', '0px');
	      $('.advices').css('text-align', 'left');
	});
	/***********/


});




});
