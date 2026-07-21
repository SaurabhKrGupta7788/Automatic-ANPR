// const socket = new WebSocket('ws://' + window.location.host + '/ws/dashboard/');

// socket.onmessage = function(e) {
//     const data = JSON.parse(e.data);
    
//     // Update Inspector Panel
//     document.getElementById('inspector-image').src = data.img_path;
//     document.getElementById('plate-field').value = data.plate;
//     const bar = document.getElementById('confidence-bar');
//     bar.style.width = data.confidence + '%';
//     if (data.confidence < 80) {
//         bar.classList.add('bg-orange-500');
//         bar.classList.add('animate-pulse');
//     } else {
//         bar.classList.remove('bg-orange-500', 'animate-pulse');
//         bar.classList.add('bg-green-500');
//     }

//     // Add to Journey Timeline
//     const card = `
//         <div class="card flex gap-2 bg-gray-800 border-l-4 border-green-500 p-2 w-64 shrink-0">
//             <img src="${data.img_path}" class="h-12 w-12 rounded">
//             <div>
//                 <p>${data.time} | ${data.plate}</p>
//                 <p>${data.type} | ${data.color}</p>
//             </div>
//         </div>
//     `;
//     const timeline = document.getElementById('journey-timeline');
//     timeline.insertAdjacentHTML('afterbegin', card);

//     // Limit to 10 cards
//     if (timeline.children.length > 10) {
//         timeline.removeChild(timeline.lastChild);
//     }
// };

// // GAN Toggle (Placeholder - Later integrate with backend)
// document.getElementById('gan-toggle').addEventListener('change', function() {
//     console.log('GAN Enhancement Toggled');
// });



