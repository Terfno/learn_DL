INAME:=terfno/dlfs
CNAME:=delefrsc

dev:
	@docker run -it python:3.8.2-alpine3.11 sh

run:
	@docker run -it --rm --name=${CNAME} -v ${PWD}:/py -w /py ${INAME} sh

# rm
rm:
	docker rm ${CNAME}

rmi:
	docker rmi ${INAME}
