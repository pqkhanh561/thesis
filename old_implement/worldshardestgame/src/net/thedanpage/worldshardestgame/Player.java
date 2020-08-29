package net.thedanpage.worldshardestgame;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.Shape;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.concurrent.TimeUnit;

public class Player {

	/** The X coordinate of the player. */
	private int x;

	/** The Y coordinate of the player. */
	private int y;

	private int dem = 0;

	public int win = 0;

	/**
	 * The X coordinate of the player, snapped to the grid of 40x40 tiles.
	 * snapX = x/40
	 */
	private int snapX;

	/**
	 * The Y coordinate of the player, snapped to the grid of 40x40 tiles.
	 * snapY = y/40
	 */
	private int snapY;

	/** True if the player is colliding with a tile above them. */
	private boolean collidingUp;

	/** True if the player is colliding with a tile below them. */
	private boolean collidingDown;

	/** True if the player is colliding with a tile to their left. */
	private boolean collidingLeft;

	/** True if the player is colliding with a tile to their right. */
	private boolean collidingRight;

	/** The smack sound. */

	/** The number of times the player has died. */
	private int deaths;

	/** True if the player has been hit and is not allowed to move. */
	public boolean dead;

	/** The opacity of the player. */
	private double opacity;

	public String key;

	public Player() {
		this.x = 400;
		this.y = 300;
		this.snapX = x/40;
		this.snapY = y/40;
		this.collidingUp = false;
		this.collidingDown = false;
		this.collidingLeft = false;
		this.collidingRight = false;
		this.deaths = 0;
		this.dead = false;
		this.opacity = 255;
		this.key = null;
	}





	public Player(int x, int y) {
		this.x = x;
		this.y = y;
		this.snapX = x/40;
		this.snapY = y/40;
		this.collidingUp = false;
		this.collidingDown = false;
		this.collidingLeft = false;
		this.collidingRight = false;
		this.deaths = 0;
		this.dead = false;
		this.opacity = 255;
		this.key = null;
	}





	public void draw(Graphics g) {
		g.setColor(new Color(0, 0, 0, (int) opacity));
		g.fillRect(x - 15, y - 15 + 22, 31, 31);
		g.setColor(new Color(255, 0, 0, (int) opacity));
		g.fillRect(x-12, y-12 + 22,
				25, 25);
	}





	Tile getRelativeTile(GameLevel level, int x1, int y1, int xOff, int yOff) {
		for (Tile t : level.getTileMap()) {
			if (x1/40 + xOff == t.getSnapX() && y1/40 + yOff == t.getSnapY()) {
				return t;
			}
		}
		return null;
	}





	Tile getTile(GameLevel level) {
		for (Tile t : level.getTileMap()) {
			if (this.x/40 == t.getSnapX() && this.y/40 == t.getSnapY()) {
				return t;
			}
		}
		return null;
	}





	boolean doesIntersect(Rectangle a, Rectangle b) {
		return (a.x + a.width < b.x || a.x > b.x + b.width
				|| a.y + a.height < b.y || a.y > b.y + b.height);
	}





	public Rectangle getBounds() {
		return new Rectangle(this.x - 15, this.y - 15, 31, 31);
	}




	void checkCollisionUp(GameLevel level) {
		if (getRelativeTile(level, this.x - 12, this.y + 12, 0, -1) != null &&
				getRelativeTile(level, this.x - 12, this.y + 12, 0, -1).getType() == 0 ||
				getRelativeTile(level, this.x + 12, this.y + 12, 0, -1) != null &&
				getRelativeTile(level, this.x + 12, this.y + 12, 0, -1).getType() == 0) {
			this.collidingUp = true;
			return;
				}
		this.collidingUp = false;
	}





	void checkCollisionDown(GameLevel level) {
		if (getRelativeTile(level, this.x - 12, this.y - 12, 0, 1) != null &&
				getRelativeTile(level, this.x - 12, this.y - 12, 0, 1).getType() == 0 ||
				getRelativeTile(level, this.x + 12, this.y - 12, 0, 1) != null &&
				getRelativeTile(level, this.x + 12, this.y - 12, 0, 1).getType() == 0) {
			this.collidingDown = true;
			return;
				}
		this.collidingDown = false;
	}





	void checkCollisionLeft(GameLevel level) {
		if (getRelativeTile(level, this.x + 12, this.y - 12, -1, 0) != null &&
				getRelativeTile(level, this.x + 12, this.y - 12, -1, 0).getType() == 0 ||
				getRelativeTile(level, this.x + 12, this.y + 12, -1, 0) != null &&
				getRelativeTile(level, this.x + 12, this.y + 12, -1, 0).getType() == 0) {
			this.collidingLeft = true;
			return;
				}
		this.collidingLeft = false;
	}





	void checkCollisionRight(GameLevel level) {
		if (getRelativeTile(level, this.x - 12, this.y - 12, 1, 0) != null &&
				getRelativeTile(level, this.x - 12, this.y - 12, 1, 0).getType() == 0 ||
				getRelativeTile(level, this.x - 12, this.y + 12, 1, 0) != null &&
				getRelativeTile(level, this.x - 12, this.y + 12, 1, 0).getType() == 0) {
			this.collidingRight = true;
			return;
				}
		this.collidingRight = false;
	}





	void respawn(GameLevel level) {
		this.x = level.getSpawnPoint().x;
		this.y = level.getSpawnPoint().y;
		if (level.coins != null) {
			for (Coin coin : level.coins) coin.collected = false;
		}
	}





	boolean collidesWith(Shape other) {
		return this.getBounds().getBounds2D().intersects(other.getBounds2D());
	}





	public void update(GameLevel level) {
		this.snapX = this.x / 40;
		this.snapY = this.y / 40;

		if (level.coins != null) {
			for (Coin coin : level.coins) {
				if (this.collidesWith(coin.getBounds()) && !coin.collected) {
					coin.collected = true;

					//Coin sound
				}
			}
		}

		if (level.getTileMap() != new ArrayList<Tile>()) {

			if (level.allCoinsCollected()) {

				for (Tile t : level.getTileMap()) {

					if (t.getType() == 3 && this.collidesWith(t.getBounds())) {
						level.init(this, Game.levelNum);
						this.win++;
						System.out.println(this.win);
					}
				}
			}
		}

		checkCollisionUp(level);
		checkCollisionDown(level);
		checkCollisionLeft(level);
		checkCollisionRight(level);

		if (key!=null && key.trim().equalsIgnoreCase("reset")){
			this.dead=true;
			this.opacity=0;
		}         
		if (this.dead) {
			this.opacity -= 255/75;

			if (this.opacity < 0) this.opacity = 0;

			if (this.opacity == 0) {
				this.dead = false;
				this.opacity = 255;
				this.respawn(level);
			}
		} else {
			if (Input.up.isPressed && !this.collidingUp) this.y -=40;
			if (Input.down.isPressed && !this.collidingDown) this.y +=40;
			if (Input.left.isPressed && !this.collidingLeft){
				this.x -=40;
			}             
			if (Input.right.isPressed && !this.collidingRight) this.x +=40;

			//Input from file
			if (key != null){
				if (key.trim().equalsIgnoreCase("up") && !this.collidingUp) this.y -=40;
				if (key.trim().equalsIgnoreCase("down") && !this.collidingDown) this.y +=40;
				if (key.trim().equalsIgnoreCase("left") && !this.collidingLeft) this.x -=40;
				if (key.trim().equalsIgnoreCase("right") && !this.collidingRight) this.x +=40;
			}
		}

		if (this.x > 800) this.x = 0;
		if (this.x < 0) this.x = 800;
		if (this.y > 600) this.y = 0;
		if (this.y < 0) this.y = 600;

		if (!this.dead) {
			for (Dot dot : level.dots) {
				//if (((int)((this.x-12)/40)== dot.getSnapX()) && ((int)((this.y+12)/40) == dot.getSnapY())) {
				if (this.collidesWith(dot.getBounds())) {
					this.deaths ++;
					this.dead = true;
				}
				}
			}
		}





		public int getX() {
			return this.x;
		}



		public int getY() {
			return this.y;
		}



		public int getSnapX() {
			return this.snapX;
		}



		public int getSnapY() {
			return this.snapY;
		}



		public int getWidth() {
			return (int) this.getBounds().getWidth();
		}



		public int getHeight() {
			return (int) this.getBounds().getHeight();
		}



		public boolean isCollidingLeft() {
			return this.collidingLeft;
		}



		public boolean isCollidingRight() {
			return this.collidingRight;
		}



		public boolean isCollidingUp() {
			return this.collidingUp;
		}



		public boolean isCollidingDown() {
			return this.collidingDown;
		}



		public int getDeaths() {
			return this.deaths;
		}



		public boolean isDead() {
			return this.dead;
		}



		public void setDead(boolean dead) {
			this.dead = dead;
		}



		public double getOpacity() {
			return this.opacity;
		}



		public void reset() {
			this.x = 400;
			this.y = 300;
			this.snapX = x/40;
			this.snapY = y/40;
			this.collidingUp = false;
			this.collidingDown = false;
			this.collidingLeft = false;
			this.collidingRight = false;
			this.deaths = 0;
			this.dead = false;
			this.opacity = 255;
		}





		@Override
		public String toString() {
			return "Player [x=" + x + ", y=" + y + ", snapX=" + snapX + ", snapY="
				+ snapY + ", collidingUp=" + collidingUp + ", collidingDown="
				+ collidingDown + ", collidingLeft=" + collidingLeft
				+ ", collidingRight=" + collidingRight + ", deaths=" + deaths
				+ ", dead=" + dead + "]";
		}
	}
