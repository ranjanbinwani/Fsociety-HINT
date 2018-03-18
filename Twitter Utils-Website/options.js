$('.btn').hide();
$(document).ready(function() {
	
	$('#loading').fadeOut(3500);
	$('.btn').show();
	$('.btn:first-child').delay(400).queue(function(next) {
		$(this).addClass('hover').delay(1800).queue(function(next) {
			$(this).removeClass('hover');
		});
		next();
	});
});