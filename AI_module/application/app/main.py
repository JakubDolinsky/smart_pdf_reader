from AI_module.infra_layer import check_db_ready

#from app.server import run_server

def main() -> None:
    # check db state
    check_db_ready()

    # run application
    #run_server()


if __name__ == "__main__":
    main()