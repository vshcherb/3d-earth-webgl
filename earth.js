// +   TODO device independent constant vs pixel-perfect - check!
// +   TODO proper tile scale
// + ? TODO autochange cam height on panning (different projection)
// TODO LOD of details setting!

// TODO precise drag/drop movements 
// TODO Hangs around poles
// TODO autoload tiles on tilt / on pan

// TODO Ellipse earth: 1) getDistance 2) problem with rotation (camera height)
// TODO holes between inequal tiles (should be rectangular network)
// TODO Day/night shadow on Earth

// ! OPENGL allows only texture power of 2 !
const TilesCanvasSize = 8; // 8 -> 7x7 (1st row taken by ice / empty)
// let GlCanvasSize = 1024; // < TilesCanvasSize * TileSize
const GlCanvasSize = 768;

const RotateAroundCenter = true;

// Global CONSTANTS
// const EarthRadiusEquator = 6378.137;
const EarthRadiusEquator = 6356.752;
const EarthRadiusPolar = 6356.752;
const EarthSkew = EarthRadiusPolar / EarthRadiusEquator; // 0.997; // earth skew
const MIN_LATITUDE = -85.0511;
const MAX_LATITUDE = 85.0511;
const LATITUDE_TURN = 180.0;
const MIN_LONGITUDE = -180.0;
const MAX_LONGITUDE = 180.0;
const LONGITUDE_TURN = 360.0;

const DELAY_TO_REPLACE_FRESH_TEXTURE = 1000;
const DELAY_TO_REPLACE_PARTIAL_TEXTURE = 250;

// Tiles loading
const TileURL = 'https://tile.openstreetmap.org';
const TileExtension = '.png';
// const TileURL = 'tiles';
// const TileExtension = '.png.tile';
const TileMinZoom = 1;
const TileMaxZoom = 18;
const TileSize = 256;

// 2 - Plane z Near set in the middle between camera [0, eyePosition/ EarthRadiusEquator, 0] and Earth Look [0, 1, 0] 
const PlaneZNear = 2; 

const CONFIG = {
    loadTexture: true,
    updateBuffer: false,

    syncZoom: true,
    tilesOnAndGridOff: true,

    cameraHeight: 20000, // insync with eyeZoom
    cameraZoom : 2.5,

    cameraAngle: 0,
    cameraLat: 52.37313,
    cameraLon: 4.89875,

    targetLat: 0,
    targetLon: 0,
    targetDist: 0,
    
    rotLatSpeed: 0,
    rotLatDir: 1,
    rotLonSpeed: 0,

    textureTilesZoom: 4,
    textureTilesBbox: {sx: 0, sy: 0, w: TilesCanvasSize, h: TilesCanvasSize},
    uploadedTextureMeta: { sx: 0, sy: 0, w: TilesCanvasSize, h: TilesCanvasSize, z: 4 },
    uploadedTexture: null,
    
    // global vertices rendering
    minVertexZoom: 6,
    vertexSpiral: 2, // ?? related to TilesCanvasSize GlCanvasSize (so vertex zoom / tiles displayed enough)??

    vertexZoom: 5,
    drawMode: 'TRIANGLES',
    
    fieldOfView: 60, // degrees up to 180
};

main();
addListeners();

// input: h in [0,360] and s,v in [0,1] - output: r,g,b in [0,1]
function hsv2rgb(h, s, v, a) {
    let f = (n, k = (n + h / 60) % 6) => v - v * s * Math.max(Math.min(k, 4 - k, 1), 0);
    return [f(5), f(3), f(1), a];
}   


function checkLongitude(longitude) {
    if (longitude >= MIN_LONGITUDE && longitude <= MAX_LONGITUDE) {
        return longitude;
    }
    while (longitude <= MIN_LONGITUDE || longitude > MAX_LONGITUDE) {
        if (longitude < 0) {
            longitude += LONGITUDE_TURN;
        } else {
            longitude -= LONGITUDE_TURN;
        }
    }
    return longitude;
}

function checkLatitude(latitude) {
    if (latitude >= MIN_LATITUDE && latitude <= MAX_LATITUDE) {
        return latitude;
    }
    while (latitude < -90 || latitude > 90) {
        if (latitude < 0) {
            latitude += LATITUDE_TURN;
        } else {
            latitude -= LATITUDE_TURN;
        }
    }
    if (latitude < MIN_LATITUDE) {
        return MIN_LATITUDE;
    } else if (latitude > MAX_LATITUDE) {
        return MAX_LATITUDE;
    }
    return latitude;
}


function getDistance(lat1, lon1, lat2, lon2) {
    let dLat = checkLatitude(lat2 - lat1) / 180.0 * Math.PI;
    let dLon = checkLongitude(lon2 - lon1) / 180.0 * Math.PI;
    let a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(lat1 / 180.0 * Math.PI) * Math.cos(lat2 / 180.0 * Math.PI) *
        Math.sin(dLon / 2 / 180.0 * Math.PI) * Math.sin(dLon / 2 / 180.0 * Math.PI);
    return (2 * EarthRadiusEquator * 1000 * Math.asin(Math.sqrt(a)));
}
function getTileNumberY(zoom, latitude) {
    latitude = checkLatitude(latitude) / 180.0 * Math.PI;
	let eval = Math.log(Math.tan(latitude) + 1 / Math.cos(latitude));
    return (1 - eval / Math.PI) / 2 * getPowZoom(zoom);
}

function getTileNumberX(zoom, longitude) {
    longitude = checkLongitude(longitude);
	const powZoom = getPowZoom(zoom);
    let dz = (longitude + 180.0) /360.0 * powZoom;
    if (dz >= powZoom) {
        return powZoom - 0.01;
    }
    return dz;
}

function getLatitudeFromTile(zoom, y) {
    let sign = y < 0 ? -1 : 1;
    return Math.atan(sign * Math.sinh(Math.PI * (1 - 2 * y / getPowZoom(zoom)))) * 180 / Math.PI;
}
function getLongitudeFromTile(zoom, x) {
    return x / getPowZoom(zoom) * 360.0 - 180.0;
}

function getPowZoom(zoom) {
    if (zoom >= 0 && zoom - Math.floor(zoom) < 0.001) {
        return 1 << parseInt(zoom);
    } else {
        return Math.pow(2, zoom);
    }
}

// Start here
function main() {
    // GlCanvasSize = document.body.clientWidth / 2 > 1024 ? 1024 : 512ж
    const canvas = document.querySelector("#glcanvas");
    canvas.width = GlCanvasSize;
    canvas.height = GlCanvasSize;
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    // If we don't have a GL context, give up now
    if (!gl) {
        alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        return;
    }

    // Vertex shader program
    const vsSource = `
    attribute vec4 aVertexPosition;
    attribute vec4 aVertexColor;
    attribute vec2 aTextureCoord;

    uniform mat4 uProjectionMatrix;
    uniform mat4 uRotationMatrix;
    varying lowp vec4 vColor;
    varying highp vec2 vTextureCoord;

    void main(void) {
      gl_Position = uProjectionMatrix * uRotationMatrix * aVertexPosition;
      vColor = aVertexColor;
      vTextureCoord = aTextureCoord;
    }
  `;

    // Fragment shader program
    const fsSource = `
    varying lowp vec4 vColor;
    varying highp vec2 vTextureCoord;
    uniform sampler2D uSampler;
    uniform lowp int uColor;
    uniform lowp int uTexture;

    void main(void) {
        gl_FragColor = vec4(uColor) * vColor 
                    + vec4(uTexture) * texture2D(uSampler, vTextureCoord); // tiles
    }
  `;

    // Initialize a shader program; this is where all the lighting
    // for the vertices and so forth is established.
    const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

    // Collect all the info needed to use the shader program.
    // Look up which attributes our shader program is using
    // for aVertexPosition, aVertexColor and also
    // look up uniform locations.
    const programInfo = {
        program: shaderProgram,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(shaderProgram, "aVertexPosition"),
            vertexColor: gl.getAttribLocation(shaderProgram, "aVertexColor"),
            textureCoord: gl.getAttribLocation(shaderProgram, "aTextureCoord"),
        },
        uniformLocations: {
            projectionMatrix: gl.getUniformLocation(shaderProgram, "uProjectionMatrix"),
            rotationMatrix: gl.getUniformLocation(shaderProgram, "uRotationMatrix"),
            uSampler: gl.getUniformLocation(shaderProgram, "uSampler"),
            uColor: gl.getUniformLocation(shaderProgram, "uColor"),
            uTexture: gl.getUniformLocation(shaderProgram, "uTexture"),
        },
    };
    // Here's where we call the routine that builds all the
    // objects we'll be drawing.
    var buffers = initBuffers(gl);
    // https://tile.openstreetmap.org/3/4/2.png
    // Browsers copy pixels from the loaded image in top-to-bottom order —
    // from the top-left corner; but WebGL wants the pixels in bottom-to-top
    // order — starting from the bottom-left corner. So in order to prevent
    // the resulting image texture from having the wrong orientation when
    // rendered, we need to make the following call, to cause the pixels to
    // be flipped into the bottom-to-top order that WebGL expects.

    
    var then = 0;
    // Draw the scene repeatedly
    function render(now) {
        if (CONFIG.loadTexture) {
            loadTilesTexture(gl);
            CONFIG.loadTexture = false;
            buffers = initBuffers(gl);
            CONFIG.updateBuffer = false;
        }
        if (CONFIG.updateBuffer) {
            buffers = initBuffers(gl);
            CONFIG.updateBuffer = false;
        }
        const deltaTime = now - then;
        then = now;
        drawScene(gl, programInfo, buffers, deltaTime / 1000.0, CONFIG.uploadedTexture);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

// Initialize the buffers we'll need.
function initBuffers(gl) {
    // Now set up the colors for the faces. We'll use solid colors
    // for each face.
    const z = Math.max(CONFIG.minVertexZoom, CONFIG.vertexZoom);    
    const faceColors = [];
    const faceColorsCount = 32;
    for (var cind = 0; cind < faceColorsCount; cind++) {
        faceColors.push(hsv2rgb((360 / faceColorsCount) * cind, 0.9, 0.9, 1));
    }
    var vind = 0;
    var vertCount = 0;
    const texStepX = 1 / CONFIG.textureTilesBbox.w;
    const texStepY = 1 / CONFIG.textureTilesBbox.h;
    
    let cx = Math.floor(getTileNumberX(z, CONFIG.targetLon)); //cameraLon
    let cy = Math.floor(getTileNumberY(z, CONFIG.targetLat)); //cameraLat
    // Now create an array of positions for the cube.
    let colors = [];
    let positions = [];
    let indices = []
    let textureCoordinates = [];
    let queue = [{ x: cx, y: cy, z: z, step: -1}];
    let qind = -1;
    let polePos = [[], []];
    const texBBox = CONFIG.uploadedTextureMeta;
    while (++qind < queue.length) {
        let tile = queue[qind];
        if (tile.x < 0 || tile.y < 0 || tile.x >= (1 << tile.z) || tile.y >= (1 << tile.z)) {
            continue;
        }
        if (tile.step != 0) {
            let nextx = tile.x + (tile.x  % 2 == 1 ? -1 : 1);
            let nexty = tile.y + (tile.y % 2 == 1 ? -1 : 1);
            // add 3 neigbhours
            queue.push({ x: tile.x, y: nexty, z: tile.z, step: 0 });
            queue.push({ x: nextx, y: tile.y, z: tile.z, step: 0});
            queue.push({ x: nextx, y: nexty, z: tile.z, step: 0});
            // add parent
            queue.push({x: Math.floor(nextx / 2), y: Math.floor(nexty / 2), z: tile.z - 1, step: 1});
            if (tile.step == 1) {
                // inner tiles already processed
                continue;
            }
        } else if (tile.z < CONFIG.minVertexZoom
               || (tile.z < z && Math.abs((cx >> (z - tile.z)) - tile.x) <= CONFIG.vertexSpiral
                              && Math.abs((cy >> (z - tile.z)) - tile.y) <= CONFIG.vertexSpiral)) {
            queue.push({ x: tile.x * 2, y: tile.y * 2, z: tile.z + 1, step: 0 });
            queue.push({ x: tile.x * 2 + 1, y: tile.y * 2, z: tile.z + 1, step: 0 });
            queue.push({ x: tile.x * 2, y: tile.y * 2 + 1, z: tile.z + 1, step: 0 });
            queue.push({ x: tile.x * 2 + 1, y: tile.y * 2 + 1, z: tile.z + 1, step: 0 });
            continue;
        }
        
        // GEO: geolatitude = 90 - lat, geolongitude = lon - 180
        const latt = Math.PI / 2 - getLatitudeFromTile(tile.z, tile.y) / (180 / Math.PI);
        const latb = Math.PI / 2 - getLatitudeFromTile(tile.z, tile.y + 1) / (180 / Math.PI);
        const lonl = getLongitudeFromTile(tile.z, tile.x) / (180 / Math.PI) + Math.PI;
        const lonr = getLongitudeFromTile(tile.z, tile.x + 1) / (180 / Math.PI) + Math.PI;
        positions.push(Math.sin(lonl) * Math.sin(latt), EarthSkew * Math.cos(latt), Math.cos(lonl) * Math.sin(latt));
        positions.push(Math.sin(lonl) * Math.sin(latb), EarthSkew * Math.cos(latb), Math.cos(lonl) * Math.sin(latb));
        positions.push(Math.sin(lonr) * Math.sin(latt), EarthSkew * Math.cos(latt), Math.cos(lonr) * Math.sin(latt));
        positions.push(Math.sin(lonr) * Math.sin(latb), EarthSkew * Math.cos(latb), Math.cos(lonr) * Math.sin(latb));
        
        const poslen = positions.length;
        if (tile.y == 0) {
            polePos[0].push(positions[poslen - 12], positions[poslen - 11], positions[poslen - 10]);
            polePos[0].push(positions[poslen - 6], positions[poslen - 5], positions[poslen - 4]);
            polePos[0].push(0, EarthSkew, 0);
        } 
        if (tile.y + 1 == (1 << tile.z)) {
            polePos[1].push(positions[poslen - 9], positions[poslen - 8], positions[poslen - 7]);
            polePos[1].push(positions[poslen - 3], positions[poslen - 2], positions[poslen - 1]);
            polePos[1].push(0, -EarthSkew, 0);
        }
        

        
        let leftTex = (tile.x / getPowZoom(tile.z - texBBox.z) - texBBox.sx) / texBBox.w  + texStepX;
        let rightTex = ((tile.x + 1) / getPowZoom(tile.z - texBBox.z) - texBBox.sx) / texBBox.w + texStepX;
        let topTex = ((tile.y) / getPowZoom(tile.z - texBBox.z) - texBBox.sy) / texBBox.h + texStepY;
        let bottomTex = ((tile.y + 1) / getPowZoom(tile.z - texBBox.z) - texBBox.sy) / texBBox.h + texStepY;
        if (leftTex < 0 || rightTex > 1 || topTex < 0 || bottomTex > 1) {
            leftTex = 0; rightTex = texStepX;
            topTex = 0; bottomTex = texStepY;
        }
        textureCoordinates.push(
            // 0, 0, step * i, 0, step * i, step * j, 0, step * j,
            leftTex, topTex, leftTex, bottomTex,
            rightTex, topTex, rightTex, bottomTex
        );
        indices.push(vind, vind + 1, vind + 2, vind + 2, vind + 1, vind + 3);
        vind += 4;
        vertCount += 6;
        const c = faceColors[(5 * qind % faceColors.length)];
        // const c = faceColors[Math.round(Math.random() * faceColors.length) % faceColors.length];
        // Repeat each color four times for the four vertices of the face
        colors.push(c[0], c[1], c[2], c[3]);
        colors.push(c[0], c[1], c[2], c[3]);
        colors.push(c[0], c[1], c[2], c[3]);
        colors.push(c[0], c[1], c[2], c[3]);
    }
    
    // console.log("VERTICES " + qind);
    
    for (var l = 0; l < polePos.length; l++) {
        for (let i = 0; i < polePos[l].length; i += 9) {
            for (var k = 0; k < 9; k++) {
                positions.push(polePos[l][i + k]);
            }
            indices.push(vind, vind + 1, vind + 2);
            vind += 3;
            vertCount += 3;
            // Repeat each color 3 times for the four vertices of the face
            const c = [0.9, 0.9, 0.7, 0.9];
            colors.push(c[0], c[1], c[2], c[3]);
            colors.push(c[0], c[1], c[2], c[3]);
            colors.push(c[0], c[1], c[2], c[3]);
            textureCoordinates.push(texStepX, 0, texStepX, texStepY - 0.1, 2 * texStepX, texStepY);
        }
    }
    
    // Now pass the list of positions into WebGL to build the
    // shape. We do this by creating a Float32Array from the
    // JavaScript array, then use it to fill the current buffer.

    // Create a buffer for the cube's vertex positions.
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    const textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, textureCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoordinates), gl.STATIC_DRAW);

    // Build the element array buffer; this specifies the indices
    // into the vertex arrays for each face's vertices.
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    // Now send the element array to GL
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
    // hack to display on ui 
    document.getElementById('sliderVertexZoomText').value = 'Vertex zoom: ' + CONFIG.vertexZoom + ', ' +
        vertCount;
    return {
        verticesCount: vertCount,
        textureCoord: textureCoordBuffer,
        position: positionBuffer,
        color: colorBuffer,
        indices: indexBuffer,
    };
}

//
// Draw the scene.
//
function drawScene(gl, programInfo, buffers, deltaTime, texture) {
    gl.clearColor(0.0, 0.0, 0.0, 1.0); // Clear to black, fully opaque
    gl.clearDepth(1.0); // Clear everything
    gl.enable(gl.DEPTH_TEST); // Enable depth testing
    gl.depthFunc(gl.LEQUAL); // Near things obscure far things

    // Clear the canvas before we start drawing on it.

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Create a perspective matrix, a special matrix that is
    // used to simulate the distortion of perspective in a camera.
    
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const projectionMatrix = mat4.create();

    // note: glmatrix.js always has the first argument
    // as the destination to receive the result.
   // mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
    const eye = vec3.fromValues(0, 0, - 1 - (CONFIG.cameraHeight / EarthRadiusEquator));
    
    // const lookAtAngleMax = Math.asin(1 / (1 + CONFIG.cameraHeight / EarthRadiusEquator));
    // const lookAt = vec3.fromValues(0, Math.tan(CONFIG.cameraAngle * lookAtAngleMax) * (EarthSkew + CONFIG.cameraHeight / EarthRadiusPolar) , 0);
    const lookAt = vec3.fromValues(0, Math.tan(CONFIG.cameraAngle / 180 * Math.PI) * (EarthSkew + CONFIG.cameraHeight / EarthRadiusPolar), 0);
    const lookAtMatrix = mat4.create();
    const perspectiveMatrix = mat4.create();
    
    // recalculate dynamically
    const zNear = (CONFIG.cameraHeight / EarthRadiusEquator) / PlaneZNear;
    const zFar = 100;
    mat4.lookAt(lookAtMatrix, eye, lookAt, vec3.fromValues(0, 1, 0));
    mat4.perspective(perspectiveMatrix, CONFIG.fieldOfView * Math.PI / 180, aspect, zNear, zFar);
    mat4.multiply(projectionMatrix, lookAtMatrix, projectionMatrix);
    mat4.multiply(projectionMatrix, perspectiveMatrix, projectionMatrix);

    const rotationMatrix = mat4.create();
    const zoomRotCoeef = CONFIG.cameraHeight / 8000;
    CONFIG.cameraLon += deltaTime * CONFIG.rotLonSpeed * zoomRotCoeef;
    if (CONFIG.cameraLon > 180) {
        CONFIG.cameraLon -= 360;
    }
    CONFIG.cameraLat += CONFIG.rotLatDir * deltaTime * CONFIG.rotLatSpeed * zoomRotCoeef;
    if (CONFIG.rotLatSpeed > 0 || CONFIG.rotLonSpeed > 0) {
        updateCameraPosText();
    }
    if (CONFIG.cameraLat >= 90) {
        CONFIG.cameraLat = 90;
        CONFIG.rotLatDir = -1;
    }
    if (CONFIG.cameraLat <= -90) {
        CONFIG.cameraLat = -90;
        CONFIG.rotLatDir = 1;
    }
    mat4.rotate(
        rotationMatrix, // destination matrix
        rotationMatrix, // matrix to rotate
        - CONFIG.cameraLat / (180 / Math.PI), // amount to rotate in radians
        [1, 0, 0]
    ); // axis to rotate around (X) 

    mat4.rotate(
        rotationMatrix, // destination matrix
        rotationMatrix, // matrix to rotate
        - CONFIG.cameraLon / (180 / Math.PI), // amount to rotate in radians
        [0, 1, 0]
    ); // axis to rotate around (Z)

    // Tell WebGL how to pull out the positions from the position
    // buffer into the vertexPosition attribute
    {
        const numComponents = 3;
        const type = gl.FLOAT;
        const normalize = false;
        const stride = 0;
        const offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
        gl.vertexAttribPointer(
            programInfo.attribLocations.vertexPosition,
            numComponents,
            type,
            normalize,
            stride,
            offset
        );
        gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
    }

    // Tell WebGL how to pull out the colors from the color buffer
    // into the vertexColor attribute.
    {
        const numComponents = 4;
        const type = gl.FLOAT;
        const normalize = false;
        const stride = 0;
        const offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.color);
        gl.vertexAttribPointer(
            programInfo.attribLocations.vertexColor,
            numComponents,
            type,
            normalize,
            stride,
            offset
        );
        gl.enableVertexAttribArray(programInfo.attribLocations.vertexColor);
    }

    // Tell WebGL how to pull out the texture coordinates from
    // the texture coordinate buffer into the textureCoord attribute.
    {
        const numComponents = 2;
        const type = gl.FLOAT;
        const normalize = false;
        const stride = 0;
        const offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.textureCoord);
        gl.vertexAttribPointer(
            programInfo.attribLocations.textureCoord,
            numComponents,
            type,
            normalize,
            stride,
            offset
        );
        gl.enableVertexAttribArray(programInfo.attribLocations.textureCoord);
    }

    // Tell WebGL which indices to use to index the vertices
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);
    // Tell WebGL to use our program when drawing
    gl.useProgram(programInfo.program);
    // Set the shader uniforms
    
    gl.uniformMatrix4fv(programInfo.uniformLocations.projectionMatrix, false, projectionMatrix);
    gl.uniformMatrix4fv(programInfo.uniformLocations.rotationMatrix, false, rotationMatrix);
    if (texture) {
        // Specify the textures
        // Tell WebGL we want to affect texture unit 0
        gl.activeTexture(gl.TEXTURE0);
        // // Bind the texture to texture unit 0
        gl.bindTexture(gl.TEXTURE_2D, texture);
    }
    // Tell the shader we bound the texture to texture unit 0
    gl.uniform1i(programInfo.uniformLocations.uSampler, 0);

    
    gl.uniform1i(programInfo.uniformLocations.uColor, CONFIG.tilesOnAndGridOff ? 0 : 1);
    gl.uniform1i(programInfo.uniformLocations.uTexture, CONFIG.tilesOnAndGridOff ? 1 : 0);

    {
        const type = gl.UNSIGNED_SHORT;
        const offset = 0;
        gl.drawElements(gl[CONFIG.drawMode], buffers.verticesCount, type, offset);
    }
}

//
// Initialize a shader program, so WebGL knows how to draw our data
//
function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

    // Create the shader program
    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    // If creating the shader program failed, alert

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert(
            "Unable to initialize the shader program: " +
            gl.getProgramInfoLog(shaderProgram)
        );
        return null;
    }

    return shaderProgram;
}

//
// creates a shader of the given type, uploads the source and
// compiles it.
//
function loadShader(gl, type, source) {
    const shader = gl.createShader(type);

    // Send the source to the shader object

    gl.shaderSource(shader, source);

    // Compile the shader program

    gl.compileShader(shader);

    // See if it compiled successfully

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(
            "An error occurred compiling the shaders: " + gl.getShaderInfoLog(shader)
        );
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

let textureUploadPending = false;
function uploadTexture(gl, texture, hdcanvas, delay) {
    textureUploadPending = true;
    setTimeout(() => {
        if (!textureUploadPending) {
            return false;
        }
        textureUploadPending = false;
        const level = 0;
        const internalFormat = gl.RGBA;
        const srcFormat = gl.RGBA;
        const srcType = gl.UNSIGNED_BYTE;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, srcFormat, srcType, hdcanvas);
        // WebGL1 has different requirements for power of 2 images
        // vs non power of 2 images so check if the image is a  power of 2 in both dimensions.
        gl.generateMipmap(gl.TEXTURE_2D);
        // No, it's not a power of 2. Turn off mips and set wrapping to clamp to edge
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        CONFIG.uploadedTextureMeta = Object.assign({}, CONFIG.textureTilesBbox);
        CONFIG.uploadedTextureMeta.z = CONFIG.textureTilesZoom;
        CONFIG.uploadedTexture = texture;
        CONFIG.updateBuffer = true;
    }, delay);
}

function loadTilesTexture(gl) {
    const texture = gl.createTexture();
    const hdcanvas = document.querySelector("#hiddencanvas");
    hdcanvas.width = TilesCanvasSize * TileSize;
    // reserve 1st row for empty tiles
    hdcanvas.height = TilesCanvasSize * TileSize;
    const ctx = hdcanvas.getContext("2d");
    ctx.beginPath();
    ctx.fillStyle = "#eee";
    ctx.fillRect(0, 0, hdcanvas.width, hdcanvas.height);
    ctx.strokeStyle = "#333";
    for (var x = 0; x < hdcanvas.width; x += TileSize / 8) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, hdcanvas.height);
        ctx.stroke();
    }
    for (var y = 0; y < hdcanvas.height; y += TileSize / 8) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(hdcanvas.width, y);
        ctx.stroke();
    }
    ctx.fillStyle = "#ddd"; // Ice Pole
    ctx.fillRect(TileSize, 0, TileSize, TileSize);


    const zoom = CONFIG.textureTilesZoom;
    const maxTileId = 1 << CONFIG.textureTilesZoom;
    
    // cameraLon, cameraLat
    const startX = Math.max(0, Math.round(getTileNumberX(zoom, CONFIG.targetLon) - (TilesCanvasSize - 1) / 2 ));
    const startY = Math.max(0, Math.round(getTileNumberY(zoom, CONFIG.targetLat) - (TilesCanvasSize - 1) / 2 ));
    CONFIG.textureTilesBbox.sx = startX;
    CONFIG.textureTilesBbox.sy = startY;
    uploadTexture(gl, texture, hdcanvas, CONFIG.uploadedTexture ? 
            DELAY_TO_REPLACE_FRESH_TEXTURE : 0);

    // don't load
    //if(true) return texture;
    for (var x = 0; x < TilesCanvasSize - 1; x++) {
        if (!(x + startX < maxTileId) ) {
            continue;
        }
        for (var y = 0; y < TilesCanvasSize - 1; y++) {
            if (!(y + startY < maxTileId)) {
                continue;
            }   
            const xT = (x + 1);
            const yT = (y + 1);
            const image = new Image();
            image.onload = function () {
                if (CONFIG.textureTilesBbox.sx == startX &&
                    CONFIG.textureTilesZoom == zoom && CONFIG.textureTilesBbox.sy == startY) {
                    ctx.drawImage(image, xT * TileSize, yT * TileSize, TileSize, TileSize);            
                    uploadTexture(gl, texture, hdcanvas, DELAY_TO_REPLACE_PARTIAL_TEXTURE);
                }
            };
            image.crossOrigin = "anonymous";
            image.src = TileURL + "/" + zoom + "/" + 
                (x + startX) + "/" +  (y + startY) + TileExtension;
        }
    }   
    return texture;
}

function isPowerOf2(value) {
    return (value & (value - 1)) == 0;
}

function updateCameraPosText() {
    var latText = document.getElementById('latText');
    var lonText = document.getElementById('lonText');
    latText.value = 'LAT: ' + CONFIG.cameraLat.toFixed(5);
    lonText.value = 'LON: ' + CONFIG.cameraLon.toFixed(5);
}

function registerSlider(idParam, uiPrefix, idInput, idLabel, flagParam) {
    const slider = document.getElementById(idInput);
    const sliderText = document.getElementById(idLabel);
    slider.value = CONFIG[idParam];
    sliderText.value = uiPrefix + CONFIG[idParam].toString();
    const updateValue = function () {
        //slider.value = CONFIG[idParam].toString();
        CONFIG[idParam] = parseFloat(slider.value);
        if (flagParam) {
            CONFIG[flagParam] = true;
        }
        sliderText.value = uiPrefix + CONFIG[idParam].toString();
    };
    slider.addEventListener('input', updateValue);
    // return setter function
    return function(vl) {
        if (vl != slider.value) {
            slider.value = vl;
            updateValue();
        }
    };
}


function findPixelPerfectCameraHeightForZoom(zoom, fov, lat, lon) {
    let z = Math.max(2, Math.floor(zoom));
    const x = getTileNumberX(z, lon); // cameraLon
    const y = getTileNumberY(z, lat);
    let tileWidthKm = getDistance(getLatitudeFromTile(z, Math.floor(y)), getLongitudeFromTile(z, x), getLatitudeFromTile(z, Math.floor(y) + 1),
        getLongitudeFromTile(z, x)) / 1000;
    let tilesToFitScreen = (GlCanvasSize / TileSize);
    let screenInKm = (tileWidthKm * getPowZoom(z - zoom) * tilesToFitScreen);
    return (screenInKm / 2) / (Math.tan(fov * Math.PI / 180 / 2));
}

function findPixelPerfectZoomForCamHeight(zoom, fov, lat, lon, height) {
    // Use Newton method to not calculate reverse
    let calcZoom = zoom;
    let hVal = findPixelPerfectCameraHeightForZoom(calcZoom, fov, lat, lon) - height;
    let calcZoomNext = calcZoom;
    let hValNext = hVal;
    while (Math.abs(hValNext) > 0.01) {
        let newCalcZoom;
        if (calcZoomNext == calcZoom) {
            newCalcZoom = calcZoom + (hValNext > height ? 0.5 : -0.5);
        } else {
            newCalcZoom = calcZoomNext - hValNext * (calcZoomNext - calcZoom) / (hValNext - hVal);
        }
        hVal = hValNext;
        calcZoom = calcZoomNext;
        calcZoomNext = newCalcZoom;
        hValNext = findPixelPerfectCameraHeightForZoom(calcZoomNext, fov, lat, lon) - height;
    }
    return calcZoomNext;
}

// MAIN FORMULA between Camera - Target
// targetDiff = (targetLat - cameraLat)
// 1) cos(targetDiff) * Rad + cos(camAngle) * targetDist = camHeight + Rad
// 2) sin(targetDiff) * Rad = sin(camAngle) * targetDist
// 3) sin(camAngle + targetDiff) * Rad = sin(camAngle) * (Rad + camHeight)
// ZOOM = findPixelPerfectZoomForCamHeight(TargetDist)

function updateCameraLocWithGivenTarget() {
    // keep targetDist constant and move camera lat / height
    // SEE MAIN FORMULA between Camera - Target
    // step back to look at targetLat
    const lookAtAngleMax = Math.asin(EarthRadiusEquator / (EarthRadiusEquator + CONFIG.cameraHeight));
    let camAngle = 0; 
    let targetDiff = 1;
    while (targetDiff >= 1) {
        camAngle = CONFIG.cameraAngle / 180 * Math.PI;
        targetDiff = Math.sin(camAngle) * (CONFIG.targetDist / EarthRadiusEquator)
        if (targetDiff > 1) {
            CONFIG.cameraAngle *= 0.9;
        }
    } 
    CONFIG.cameraLat = CONFIG.targetLat - targetDiff * 180 / Math.PI;
    CONFIG.cameraHeight = CONFIG.targetDist * Math.cos(camAngle) + EarthRadiusEquator * (Math.cos(targetDiff) - 1);
}

function updateTargetLocWithGivenCamera() {
    // const lookAtAngleMax = Math.asin(EarthRadiusEquator / (EarthRadiusEquator + CONFIG.cameraHeight)) ;
    // const camAngle = Math.min(1, CONFIG.cameraAngle) * lookAtAngleMax;
    const camAngle = CONFIG.cameraAngle / 180 * Math.PI;
    // SEE MAIN FORMULA between Camera - Target
    CONFIG.targetLat = (Math.asin(Math.sin(camAngle) / EarthRadiusEquator * (CONFIG.cameraHeight + EarthRadiusEquator)) - 
                        camAngle) / Math.PI * 180 + CONFIG.cameraLat;
    const targetDiff = (CONFIG.targetLat - CONFIG.cameraLat) / 180 * Math.PI;
    // targetDist = (camHeight + Rad) * tan(targetDiff) / (sin(camAngle) + cos(camAngle) * tan(targetDiff))
    if (Math.abs(targetDiff) < 0.00001) {
        // limit if targetDiff -> 0
        CONFIG.targetDist = CONFIG.cameraHeight; 
    } else {
        CONFIG.targetDist = Math.tan(targetDiff) *
                (CONFIG.cameraHeight + EarthRadiusEquator) / (Math.sin(camAngle) + Math.cos(camAngle) * Math.tan(targetDiff))
    }
    CONFIG.targetLon = CONFIG.cameraLon;
}


function updateTargetLocText() {

    document.getElementById("targetLatText").value = "LAT " + CONFIG.targetLat.toFixed(5);
    document.getElementById("targetLonText").value = "LON " + CONFIG.targetLon.toFixed(5);
    document.getElementById("targetDistText").value = "DIST " + (CONFIG.targetDist).toFixed(4) + " km";

    const z = (CONFIG.textureTilesZoom ? CONFIG.textureTilesZoom : 2);
    document.getElementById("targetTileZoomText").value = "Z " + z;
    document.getElementById("targetTileXText").value = "X " + getTileNumberX(z, CONFIG.targetLon).toFixed(2);
    document.getElementById("targetTileYText").value = "Y " + getTileNumberY(z, CONFIG.targetLat).toFixed(2);
    document.getElementById("targetProjDistText").value = "ON GLOBE " + (getDistance(CONFIG.targetLat, CONFIG.targetLon,
        CONFIG.cameraLat, CONFIG.cameraLon) / 1000).toFixed(3) + " km";
}


function addListeners() {
    var drawMode = document.getElementById('drawMode');
    drawMode.addEventListener('change', function (e) { 
        if(drawMode.options[drawMode.selectedIndex].value == 'TILES_GRID') {
            CONFIG.drawMode = 'TRIANGLES';
            CONFIG.tilesOnAndGridOff = false;
        } else {
            CONFIG.drawMode = drawMode.options[drawMode.selectedIndex].value;
            CONFIG.tilesOnAndGridOff = true;
        }
        CONFIG.updateBuffer = true;
    });
    registerSlider('rotLonSpeed', 'ROT LON:', 'sliderRotLonSpeed', 'sliderRotLonSpeedText');
    registerSlider('rotLatSpeed', 'ROT LAT:', 'sliderRotLatSpeed', 'sliderRotLatSpeedText');

    let mouseCoords, mousedown = false;
    const canvas = document.querySelector("#glcanvas");
    function getClippedCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const clipX = x / rect.width * 2 - 1;
        const clipY = y / rect.height * -2 + 1;
        return [clipX, clipY];
    }
    
    canvas.addEventListener('mousedown', (e) => {
        mousedown = true;
        mouseCoords = getClippedCoords(e);
    });
    canvas.addEventListener('mouseup', (e) => {
        mousedown = false;
    });
    canvas.addEventListener('mousemove', (e) => {
        if (mousedown) {
            let newCoords = getClippedCoords(e);
            CONFIG.cameraLat += (mouseCoords[1] - newCoords[1]) * CONFIG.cameraHeight / 400;
            CONFIG.cameraLon += (mouseCoords[0] - newCoords[0]) * CONFIG.cameraHeight / 400; 
            updateTargetLocWithGivenCamera();

            CONFIG.cameraZoom = findPixelPerfectZoomForCamHeight(CONFIG.cameraZoom, CONFIG.fieldOfView, CONFIG.cameraLat, CONFIG.cameraLon, CONFIG.targetDist);
            camZoom.value = CONFIG.cameraZoom;
            

            updateCamZoomAngleTxt();
            updateCameraPosText();
            updateTargetLocText();
            mouseCoords = newCoords;
        }
    });
    
    

    registerSlider('minVertexZoom', 'Min Vertex Zoom:', 'sliderMinVertexZoom', 'sliderMinVertexZoomText', 'updateBuffer');
    registerSlider('vertexSpiral', 'Vertex Details Zoom:', 'sliderVertexSpiral', 'sliderVertexSpiralText', 'updateBuffer');

    const setVertexZoom = registerSlider('vertexZoom', 'Vertex Zoom:', 'sliderVertexZoom', 'sliderVertexZoomText', 'updateBuffer');
    const setTextureTilesZoom = registerSlider('textureTilesZoom', 'Tiles Zoom:', 'sliderTextureZoom', 'sliderTextureZoomText', 'loadTexture');
    
    /// CAMERA RECALCUALATION
    const camAngle = document.getElementById("sliderEyeAngle");
    const camAngleText = document.getElementById("sliderEyeAngleText");
    const camZoom = document.getElementById("sliderEyePos");
    const camZoomText = document.getElementById("sliderEyePosText");

    function updateCamZoomAngleTxt() {
        camZoomText.value = "Zoom: " + CONFIG.cameraZoom.toFixed(2) + ", " + CONFIG.cameraHeight.toFixed(CONFIG.cameraHeight < 1 ? 3 : 3) + " km" ; 
        const lookAtAngleMax = Math.asin(1 / (1 + CONFIG.cameraHeight / EarthRadiusEquator)) * 180 / Math.PI;
        // camAngleText.value = "ANGLE: " + camAngle.value + ", " + (camAngle.value *  lookAtAngleMax).toFixed(1) + "°[" + lookAtAngleMax.toFixed(1) + "°]";
        if (camAngle.value != CONFIG.cameraAngle) {
            camAngle.value = CONFIG.cameraAngle;
        }
        camAngleText.value = "ANGLE: " + CONFIG.cameraAngle.toFixed(1) + "°[" + lookAtAngleMax.toFixed(1) + "°]";
    }

    const syncZoom = document.getElementById("syncZoom");

    function syncZooms() {
        if (CONFIG.syncZoom) {
            const text = Math.max(Math.max(TileMinZoom, 2), Math.min(Math.floor(CONFIG.cameraZoom), TileMaxZoom));
            setVertexZoom(Math.max(text + 1, CONFIG.minVertexZoom));
            setTextureTilesZoom(text);
        }
    }
    syncZoom.addEventListener('change', function () {
        CONFIG.syncZoom = this.checked;
        syncZooms();
    });

    
    function setCamZoomValue(vl) {
        CONFIG.cameraZoom = vl;
        camZoom.value = CONFIG.cameraZoom;

        CONFIG.targetDist = findPixelPerfectCameraHeightForZoom(CONFIG.cameraZoom, CONFIG.fieldOfView, CONFIG.cameraLat, CONFIG.cameraLon);
        //updateTargetLocWithGivenCamera();
        updateCameraLocWithGivenTarget();
        syncZooms();

        updateCamZoomAngleTxt();
        updateCameraPosText();
        updateTargetLocText();
    }

    camZoom.value = CONFIG.cameraZoom;
    camZoom.addEventListener('input', function () {
        setCamZoomValue(parseFloat(camZoom.value));
    });
    canvas.addEventListener('wheel', (e) => {
        const delta = -(0.05 * Math.floor(e.deltaY / 4));
        setCamZoomValue(Math.max(0, Math.min(22, CONFIG.cameraZoom + delta)));
    });
    camAngle.addEventListener('input', function () {
        const lookAtAngleMax = Math.asin(EarthRadiusEquator / (EarthRadiusEquator + CONFIG.cameraHeight));
        CONFIG.cameraAngle = Math.min(camAngle.value, lookAtAngleMax * 180 / Math.PI - 0.1);
        if (RotateAroundCenter) {
            updateCameraLocWithGivenTarget();
        } else {
            updateTargetLocWithGivenCamera();
        }
        CONFIG.cameraZoom = findPixelPerfectZoomForCamHeight(CONFIG.cameraZoom, CONFIG.fieldOfView, CONFIG.cameraLat, CONFIG.cameraLon, CONFIG.targetDist);
        camZoom.value = CONFIG.cameraZoom;

        syncZooms();

        // recalculate lat / lon / distance to keep center in same place
        updateCameraPosText();
        updateCamZoomAngleTxt();
        updateTargetLocText();
    });


    const fieldOfView = document.getElementById('sliderFOVAngle');
    const fieldOfViewText = document.getElementById('sliderFOVAngleText');
    fieldOfView.value =  CONFIG.fieldOfView;
    fieldOfViewText.value = 'FOV: ' + CONFIG.fieldOfView + '°';
    fieldOfView.addEventListener('input', function () {
        CONFIG.fieldOfView = fieldOfView.value;
        setCamZoomValue(CONFIG.cameraZoom);
        fieldOfViewText.value = 'FOV: ' + CONFIG.fieldOfView + '°';
        CONFIG.updateBuffer = true;
    });
    
    // INIT
    CONFIG.cameraHeight = findPixelPerfectCameraHeightForZoom(CONFIG.cameraZoom, CONFIG.fieldOfView, CONFIG.cameraLat, CONFIG.cameraLon);
    updateTargetLocWithGivenCamera();
    setCamZoomValue(CONFIG.cameraZoom);
    // updateCameraPosText();
    // updateCamZoomAngleTxt();
    // updateTargetLocText();
    
}