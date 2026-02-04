ymaps.ready(init);

function init() {
    var geolocation = ymaps.geolocation,
        coords = [geolocation.latitude, geolocation.longitude],
        myMap = new ymaps.Map('map', {
            center: coords,
            zoom: 10
        });

    myMap.geoObjects.add(
        new ymaps.Placemark(
            coords,
            {
                balloonContentHeader: geolocation.country,
                balloonContent: geolocation.city,
                balloonContentFooter: geolocation.region
            }
        )
    );
}