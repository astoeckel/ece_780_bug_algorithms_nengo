#!/bin/bash -e

# Fetch the number of pages in the pdf
NP=`pdfinfo "$1"  | grep "Pages:" | sed 's/Pages:[ \t]*\(0-9\)*[ \t]*/\1/'`

# Fetch the title
TITLE=`pdfinfo "$1" | grep "Title:" | sed 's/Title:[ \t]*\(.*\)/\1/'`

# Create a target folder
NAME=`basename "$1" .pdf`
DIR="$NAME"_html
mkdir -p "$DIR"
mkdir -p "$DIR/pages"


# Build the HTML
HTML="$DIR/$NAME.html"
echo -e '<!DOCTYPE html>\n<html>\n\t<head>\n\t\t<title>'$TITLE'</title>' > "$HTML"
echo -e '\t\t<style>body,html{background-color:black;overflow:hidden;font-size:4vw;width:100%;height:100%;margin:0;padding:0} section{position:fixed;top:0;bottom:0;left:0;right:0;opacity:0}section:target{z-index:1}body:not(.muted) section:target{opacity:1}iframe, img, video{position:relative;left: 50%; top:50%;transform: translateX(-50%) translateY(-50%);max-height:100%;max-width:100%;background-color:white}.incremental:not(.revealed){visibility:hidden}</style>' >> "$HTML"
echo -e '\t</head>\n\t<body>' >> "$HTML"

# Convert each page to a SVG
for I in `seq $NP`; do
	echo -en '\t\t<section>' >> "$HTML"
	IP=`printf %03d $I`
	echo "Extracting page $I..."
	qpdf --empty --pages "$1" $I -- "$DIR/pages/page_$IP.pdf"
	if MEDIA=`pdftotext "$DIR/pages/page_$IP.pdf" - | grep "#!video:"`; then
		echo "Including video..."
		MEDIA=`echo "$MEDIA" | sed 's/#!video://'`
		MIME=`file -b --mime-type "$MEDIA"`
		echo -n '<video src="data:'$MIME';base64,' >> "$HTML"
		base64 -w0 "$MEDIA" >> "$HTML"
		echo -n '"></video>' >> "$HTML"
	else
		echo "Converting page $I to SVG..."
		pdf2svg "$DIR/pages/page_$IP.pdf" "$DIR/pages/page_$IP.svg"
		sed -i 's/width="[^"]*"\s*height="[^"]*"\s*//' "$DIR/pages/page_$IP.svg"
		rm "$DIR/pages/page_$IP.pdf"
		echo -n '<img src="data:image/svg+xml;base64,' >> "$HTML"
		base64 -w0 "$DIR/pages/page_$IP.svg" >> "$HTML"
		echo -n '"/>' >> "$HTML"
	fi
	echo '</section>' >> "$HTML"
done

# HTML footer
cat >> "$HTML" <<- EOM
	<script>
// Code based on https://github.com/ThomasR/minislides/blob/master/src/minislides.js

var slides, currentPageNumber, activeSlide, incremental, keyCodeNormalized, setPage, processHash,
    revealedCls = 'revealed', incrementalSelector = '.incremental',
    querySelector = 'querySelector', loc = location, doc = document, document_body;

document_body = doc.body;
slides = Array.from(doc[querySelector + 'All']('section'));

/**
* Jump to the given page (1-based) and update location hash
* @param {number} newPageNumber Should be an integer, some falsy value or Infinity
*/
setPage = function (newPageNumber) {
    if (currentPageNumber && (currentPageNumber == newPageNumber - 1)) {
        var skip_video = false;
        for (video of activeSlide.querySelectorAll("video")) {
            if (video.currentTime < video.duration) {
                video.currentTime = video.duration;
                video.pause();
                skip_video = true;
            }
        }
        if (skip_video) {
            return;
        }
    }
    currentPageNumber = Math.min(slides.length, newPageNumber || 1);
    activeSlide = slides[currentPageNumber - 1];
    for (video of document_body.querySelectorAll("video")) {
        video.currentTime = 0;
        video.pause()
    }
    for (video of activeSlide.querySelectorAll("video")) {
        video.currentTime = 0;
        video.play();
    }
    slides.map.call(activeSlide[querySelector + 'All'](incrementalSelector), function (el) {
        el.classList.remove(revealedCls);
    });
    loc.hash = currentPageNumber;
    document_body.dataset.slideId = activeSlide.dataset.id || currentPageNumber;
};

function nextSlide() {
    incremental = activeSlide[querySelector](incrementalSelector + ':not(.' + revealedCls + ')');
    if (incremental) {
        incremental.classList.add(revealedCls);
    } else {
        setPage(currentPageNumber + 1);
    }
}

function previousSlide() {
    setPage(currentPageNumber - 1);
}

window.addEventListener('click', function (e) {
    if (e.button == 0) {
        nextSlide();
        e.preventDefault();
    }
});
window.addEventListener('wheel', function (e) {
    if (e.deltaY > 0) {
        nextSlide();
        e.preventDefault();
    } else if (e.deltaY < 0) {
        previousSlide();
    }
});
window.addEventListener('keydown', function (e, preventDefault) {
    keyCodeNormalized = e.which - 32; // - 32 for better compression
    if (!keyCodeNormalized /*keyCodeNormalized == 32 - 32*/ // space
            || !(keyCodeNormalized - (34 - 32)) // pgDn
            || !(keyCodeNormalized - (39 - 32)) // right arrow
            || !(keyCodeNormalized - (40 - 32)) // down arrow
    ) {
        nextSlide();
        preventDefault = 1;
    }
    if (!(keyCodeNormalized - (33 - 32)) // pgUp
            || !(keyCodeNormalized - (37 - 32)) // left
            || !(keyCodeNormalized - (38 - 32)) // up
    ) {
	previousSlide();
        preventDefault = 1;
    }
    if (!(keyCodeNormalized - (36 - 32))) { // home
        setPage(1);
        preventDefault = 1;
    }
    if (!(keyCodeNormalized - (35 - 32))) { // end
        setPage(Infinity); // shorter than slides.length, since it gets compressed to 1/0
        preventDefault = 1;
    }
    if (preventDefault) {
        e.preventDefault();
    }
});

// set slide ids
slides.map(function (slide, i) {
    slide.id = i + 1;
});

// poll location hash
processHash = function (newPageNumber) {
    newPageNumber = loc.hash.substr(1);
    if (newPageNumber != currentPageNumber) {
        setPage(newPageNumber);
    }
};
processHash();

// fade-in presentation
document_body.classList.add('loaded');

// start polling
setInterval(processHash, 99);
	</script>
EOM
echo -e '\t\t<!-- Code based on https://github.com/ThomasR/minislides/blob/master/src/minislides.css -->' >> "$HTML"
echo -e '\t\t<!-- Generated from LaTeX beamer using Bash black magic. -->' >> "$HTML"
echo -en '\t</body>\n</html>' >> "$HTML"

echo "Cleanup."
mv "$HTML" .
rm -r "$DIR"
echo "Done writing `basename "$HTML"    `"
