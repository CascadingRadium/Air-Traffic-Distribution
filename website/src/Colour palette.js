// const chroma = require("chroma-js")

// 	function log(s) {
// 		console.log(s)
// 	}

// 	function get_palette(n=8, Lmin=5, Lmax=90, maxLoops=10000) {
// 		log("processing with parameters n=" + n + ", Lmin=" + Lmin + ", Lmax=" + Lmax + ", maxLoops=" + maxLoops);
// 		var data = new Array();
// 		for (var color in chroma.colors) {
// 			var c = chroma(color);
// 			var L = 100 * c.hsl()[2];
// 			if ((L >= Lmin) && (L <= Lmax)) {
// 				data.push(c);
// 			}
// 		}
// 		log(data.length + " initial colors");
// 		var bestScore = 0;
// 		var bestPalette = null;
// 		var generation = 0;
// 		var loop = 0;
// 		while (loop < maxLoops) {
// 			loop += 1;
// 			var colors = data.slice(0);
// 			var palette = colors.splice(Math.floor(Math.random() * colors.length), 1);
// 			while (palette.length < n) {
// 				var availableColors = new Array();
// 				colors.forEach(function (color) {
// 					var accepted = true;
// 					for (var k in palette) {
// 						if (chroma.distance(color, palette[k]) <= bestScore) {
// 							accepted = false;
// 							break;
// 						}
// 					}
// 					if (accepted) {
// 						availableColors.push(color);
// 					}
// 				});
// 				if (availableColors.length < n - palette.length) {
// 					break;
// 				}
// 				palette = palette.concat(availableColors.splice(Math.floor(Math.random() * availableColors.length), 1));
// 				colors = availableColors.slice(0);
// 			}
// 			if (palette.length == n) {
// 				generation += loop;
// 				loop = 0;
// 				bestScore = null;
// 				for (var i = 0; i < n - 1; i++) {
// 					for (var j = i + 1; j < n; j++) {
// 						var d = chroma.distance(palette[i], palette[j]);
// 						if ((bestScore === null) || (d < bestScore)) {
// 							bestScore = d;
// 						}
// 					}
// 				}
// 				bestPalette = palette.slice(0);
// 				log("generation " + generation + ": better palette found with score " + bestScore);
//                 log("Palette " + bestPalette);
// 			}
// 		}
// 		log(generation + loop + " loops done");
// 		bestPalette.sort(function (a, b) {
// 			var score_a = (a.hsl()[0] !== a.hsl()[0] ? -8 : a.hsl()[0] / 360) + a.hsl()[1] * 2 + a.hsl()[2] * 5;
// 			var score_b = (b.hsl()[0] !== b.hsl()[0] ? -8 : b.hsl()[0] / 360) + b.hsl()[1] * 2 + b.hsl()[2] * 5;
// 			return score_a - score_b;
// 		});
// 		generation += loop;
// 		var score = Math.round(bestScore * 100) / 100;
// 		var innerHTML = '<div class="media-left"><p class="is-size-2 has-text-warning">' + score + '</p><p class="is-size-7">' + data.length + ' colors</p><p class="is-size-7">' + generation + ' loops</p></div><div class="media-content">';
// 		bestPalette.forEach(function (color) {
// 			var textColor = chroma.distance(color, chroma("black"), "lab") > 1.2 * chroma.distance(color, chroma("white"), "lab") ? "black" : "white";
// 			innerHTML += '<div class="is-pulled-left box is-radiusless has-text-' + textColor + ' has-text-weight-bold has-text-centered" style="background-color: ' + color.hex() + '"><p>' + color.name() + '</p><p>' + color.hex() + '</p></div>';
// 			});
// 		innerHTML += '</div><div class="media-right"><button class="delete" onclick="remove_palette(this)"></button></div>';
// 		return {"total_colors": data.length, "loops_done": generation, "score": score, "palette": bestPalette, "paletteHTML": innerHTML};	
// 	}

// 	get_palette(5)


// CommonJS
var distinctColors = require('distinct-colors').default
var palette = distinctColors({count:10}) // You may pass an optional config object
console.log(palette)
