$(document).ready(function() {
    $('#sos-btn').click(function() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                var lat = position.coords.latitude;
                var lng = position.coords.longitude;
                $.ajax({
                    url: '/user/sos',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ latitude: lat, longitude: lng }),
                    success: function(response) {
                        var resultDiv = $('#sos-result');
                        resultDiv.html('<div class="alert alert-success">' + response.message + '</div>');
                        if (response.nearby_cameras && response.nearby_cameras.length > 0) {
                            var list = '<ul class="list-group mt-3">';
                            response.nearby_cameras.forEach(function(camera) {
                                list += '<li class="list-group-item">Camera ID: ' + camera.id + ', URL: ' + camera.url + ', Distance: ' + camera.distance + ' km, Incident ID: ' + camera.incident_id + '</li>';
                            });
                            list += '</ul>';
                            resultDiv.append(list);
                        } else {
                            resultDiv.append('<p>No nearby cameras found.</p>');
                        }
                    },
                    error: function(err) {
                        $('#sos-result').html('<div class="alert alert-danger">Error: ' + err.responseJSON.error + '</div>');
                    }
                });
            }, function(error) {
                $('#sos-result').html('<div class="alert alert-danger">Geolocation error: ' + error.message + '</div>');
            });
        } else {
            $('#sos-result').html('<div class="alert alert-danger">Geolocation is not supported by this browser.</div>');
        }
    });
});
