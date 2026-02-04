document.addEventListener('DOMContentLoaded', function() {
    const name = document.getElementById('EditBox_Name');
    const contacts = document.getElementById('EditBox_Connection');
    const decs = document.getElementById('EditBox_Description');
    const saveBtn = document.getElementById('save-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    var editContainer = document.getElementById("edit-container");

    saveBtn.addEventListener('click', function() {
        sendDataToServer(decs, contacts, name);
        editContainer.style.display = 'none';
    });

    cancelBtn.addEventListener('click', function() {
        window.location.reload();
        editContainer.style.display = 'none';
    });
    name.addEventListener('input', function(event) {
        editContainer.style.display = 'inline';
    });
    contacts.addEventListener('input', function(event) {
        editContainer.style.display = 'inline';
    });
    decs.addEventListener('input', function(event) {
        editContainer.style.display = 'inline';
    });
    function sendDataToServer(decs, contacts, name) {
        const url = `/process_profile?information=${encodeURIComponent(decs.innerText)}&connection=${encodeURIComponent(contacts.innerText)}&name=${encodeURIComponent(name.innerText)}`;
        fetch(url, {
            method: 'POST'
        }).then(response => {
            if (!response.ok) {
                throw new Error('Failed to send data to server');
            }
            console.log('Data sent successfully');
        }).catch(error => {
            console.error('Error sending data to server:', error.message);
        });
    }

});